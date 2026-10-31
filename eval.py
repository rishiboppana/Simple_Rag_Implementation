"""Run outputs/rag_evaluation_set_30.json under several retrieval setups and write outputs/eval_results.json + eval_summary.md.

Usage: python eval.py [experiment ...]   (default: all)
Retrieval, citation and abstention are scored automatically; answer correctness and groundedness are judged by the chat LLM.
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy  # noqa: F401  (import before threads start; huggingface_hub imports it lazily)
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag import Settings, answer_question, create_embeddings, create_llm, open_store, retrieve, source_labels

OUT = Path("outputs")
settings = Settings()
embeddings = create_embeddings(settings)
store = open_store(settings, embeddings)
reranker = CrossEncoder("BAAI/bge-reranker-base")
# a larger model grades; Qwen3-8B marked correct answers wrong
judge_settings = replace(settings, chat_model=os.getenv("JUDGE_MODEL", "Qwen/Qwen3-32B"))
ABSTAIN = "I cannot find this information"


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def baseline(question: str, k: int):
    return retrieve(question, [store], k)


def rerank(question: str, k: int, fetch: int = 20):
    candidates = retrieve(question, [store], fetch)
    scores = reranker.predict([(question, doc.page_content) for doc, _ in candidates])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:k]
    return [candidates[i] for i in order]  # keeps the original similarity score for the abstention gate


def parent_child(question: str, m: int, k: int):
    """Stage 1: best m semantic chunks (parents) via similarity; stage 2: rerank all their sub-chunks, keep k."""
    parents: dict[tuple, None] = {}
    for doc, _ in retrieve(question, [store], 20):
        parents.setdefault((doc.metadata["source_file"], doc.metadata["semantic_chunk_id"]), None)
    children: list[Document] = []
    for file, semantic_id in list(parents)[:m]:
        found = store.get(where={"$and": [{"source_file": file}, {"semantic_chunk_id": semantic_id}]})
        children += [Document(page_content=t, metadata=md) for t, md in zip(found["documents"], found["metadatas"])]
    scores = reranker.predict([(question, doc.page_content) for doc in children])
    return sorted(zip(children, map(float, scores)), key=lambda item: item[1], reverse=True)[:k]


def hyde(question: str, k: int):
    prompt = f"Write a short passage (about 80 words) from a document that would answer this question. Do not say you lack information.\nQuestion: {question}\nPassage:"
    passage = strip_think(create_llm(settings).invoke(prompt).content)
    # the abstention gate still uses the real question's similarity, since hypothetical-passage scores are not comparable
    gate = retrieve(question, [store], 1)[0][1]
    hits = store.similarity_search_by_vector_with_relevance_scores(embeddings.client.embed_query(passage), k=k)
    return [(doc, gate) for doc, _ in hits]


EXPERIMENTS = {
    "baseline_k4": lambda q: baseline(q, 4),
    "k2": lambda q: baseline(q, 2),
    "k8": lambda q: baseline(q, 8),
    "rerank_k4": lambda q: rerank(q, 4),
    "hyde_k4": lambda q: hyde(q, 4),
    "rerank_k8": lambda q: rerank(q, 8),
    "rerank_k4_nogate": lambda q: rerank(q, 4),
    "rerank_k8_nogate": lambda q: rerank(q, 8),
    "parent_m1_k4_nogate": lambda q: parent_child(q, 1, 4),
    "parent_m3_k4_nogate": lambda q: parent_child(q, 3, 4),
    "parent_m3_k8_nogate": lambda q: parent_child(q, 3, 8),
    "parent_m5_k8_nogate": lambda q: parent_child(q, 5, 8),
}


def judge(case: dict, answer: str, results) -> dict:
    prompt = f"""You are grading a RAG answer. Reply with JSON only: {{"answer": 0|1|2, "grounded": 0|1}}
answer: 2 = correct and complete versus the reference; 1 = correct but materially incomplete; 0 = wrong, unsupported, or a refusal.
grounded: 1 if every factual claim in the answer is supported by the retrieved context; 0 otherwise.

Question: {case['question']}
Reference answer: {case['expected_answer']}
Retrieved context:
{chr(10).join(doc.page_content for doc, _ in results)}

System answer: {answer}
JSON:"""
    text = strip_think(create_llm(judge_settings).invoke(prompt).content)
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0))
        return {"answer": int(data["answer"]), "grounded": int(data["grounded"])}
    except Exception:
        return {"answer": 0, "grounded": 0, "judge_error": text[:200]}


def run_case(case: dict, retriever, name: str) -> dict:
    results = retriever(case["question"])
    run_settings = replace(settings, min_relevance_score=0.0) if name.endswith("nogate") else settings  # nogate: the LLM alone decides to abstain
    answer = answer_question(case["question"], [], results, run_settings)
    sources = source_labels(results)
    row = {"id": case["id"], "type": case["type"], "answer": answer, "sources": sources, "abstained": answer.startswith(ABSTAIN)}
    if case["type"] == "unanswerable":
        row["abstain_ok"] = int(row["abstained"])
        return row
    exp = case["expected_source"]
    row["retrieval"] = int(any(d.metadata["source_file"] == exp["file"] and d.metadata["page"] + 1 == exp["pdf_page"] for d, _ in results))
    row["citation"] = int(not row["abstained"] and f"{exp['file']} — page {exp['pdf_page']}" in sources)
    row.update({"answer_score": 0, "grounded": 1} if row["abstained"] else dict(zip(("answer_score", "grounded"), judge(case, answer, results).values())))
    return row


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] == "answerable"]
    cited = [r for r in ans if not r["abstained"]]
    una = [r for r in rows if r["type"] == "unanswerable"]
    return {
        "recall": sum(r["retrieval"] for r in ans) / len(ans),
        "correctness": sum(r["answer_score"] for r in ans) / (2 * len(ans)),
        "citation": sum(r["citation"] for r in cited) / max(len(cited), 1),
        "groundedness": sum(r["grounded"] for r in ans) / len(ans),
        "abstention": sum(r["abstain_ok"] for r in una) / len(una),
        "unjustified_refusals": len(ans) - len(cited),
    }


if __name__ == "__main__":
    cases = json.loads((OUT / "rag_evaluation_set_30.json").read_text())["cases"]
    names = sys.argv[1:] or list(EXPERIMENTS)
    results_path = OUT / "eval_results.json"
    all_runs = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in names:
        with ThreadPoolExecutor(4) as pool:
            rows = list(pool.map(lambda c: run_case(c, EXPERIMENTS[name], name), cases))
        all_runs[name] = {"summary": summarize(rows), "rows": rows}
        print(name, {k: round(v, 3) for k, v in all_runs[name]["summary"].items()}, flush=True)
        results_path.write_text(json.dumps(all_runs, indent=1))

    config = f"embedding={settings.embedding_model}; chunking=semantic + 800/120 tokens; LLM={settings.chat_model}; judge={judge_settings.chat_model}; min_relevance={settings.min_relevance_score}"
    lines = [f"# Evaluation summary\n\n{config}\n", "| Experiment | Recall@k | Correctness | Citation | Groundedness | Abstention | Unjustified refusals |", "|---|---|---|---|---|---|---|"]
    for name, run in all_runs.items():
        s = run["summary"]
        lines.append(f"| {name} | {s['recall']:.0%} | {s['correctness']:.0%} | {s['citation']:.0%} | {s['groundedness']:.0%} | {s['abstention']:.0%} | {s['unjustified_refusals']} |")
    (OUT / "eval_summary.md").write_text("\n".join(lines) + "\n")
