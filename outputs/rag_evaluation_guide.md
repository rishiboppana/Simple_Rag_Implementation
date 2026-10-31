# RAG Evaluation Guide — 30 Questions

## Purpose

This evaluation set measures the document-only RAG application using the two currently supplied PDFs. It is a small, project-specific validation set, not an external benchmark.

- **24 answerable cases:** Test retrieval, grounded generation, and citation.
- **6 unanswerable cases:** Test the required strict response when the documents do not support an answer.
- **Machine-readable data:** Use [`rag_evaluation_set_30.json`](rag_evaluation_set_30.json) as the source of truth for automated runs.

## How to run each case

1. Submit the question to the RAG app with both PDFs in the base collection.
2. Record the top four retrieved chunks, final answer, and displayed source references.
3. Score the result with the rubric below.
4. Run the same set after the similarity baseline, a reranker experiment, and a HyDE experiment. Do not change the questions between runs.

## Metrics

| Metric | Formula | What it verifies | Initial target |
|---|---|---|---:|
| Retrieval Recall@4 | Answerable cases where expected file/page appears in top 4 ÷ 24 | The retriever finds relevant evidence | ≥ 85% |
| Answer correctness | Sum of answer scores ÷ (24 × 2) | The generated answer contains the expected substance | ≥ 80% |
| Citation accuracy | Answerable responses with a correct displayed file/page or table source ÷ cited answerable responses | The source line is trustworthy | ≥ 95% |
| Groundedness | Answerable responses making no unsupported factual claim ÷ 24 | The model stays within retrieved evidence | ≥ 90% |
| Abstention accuracy | Correct strict responses on unanswerable cases ÷ 6 | The model refuses unsupported answers | ≥ 83% (5/6) |

Do not rely only on one combined score. A high answer score with poor citation accuracy or poor abstention is not acceptable for this document-only RAG system.

## Per-question scoring rubric

### Answerable cases

| Field | Score | Rule |
|---|---:|---|
| Retrieval | 1 / 0 | 1 if the expected file and PDF page occur in the top four retrieved chunks; otherwise 0. |
| Answer | 2 / 1 / 0 | 2 = correct and complete; 1 = correct but materially incomplete; 0 = wrong, unsupported, or an unjustified refusal. |
| Citation | 1 / 0 | 1 if the final displayed source identifies the correct document and page; otherwise 0. |
| Groundedness | 1 / 0 | 1 if no factual claim extends beyond the retrieved document evidence; otherwise 0. |

### Unanswerable cases

Score **1** only when the application says the information is unavailable in the provided documents and does not invent an answer. A labeled “best guess” is acceptable only when it is explicitly framed as an inference from retrieved evidence; for these six cases, no such evidence should be claimed.

## Question inventory

| ID | Type | Question | Expected source |
|---|---|---|---|
| C01 | Answerable | Who is the curriculum designed for? | Curriculum, PDF p. 1 |
| C02 | Answerable | What prior background is recommended? | Curriculum, PDF p. 1 |
| C03 | Answerable | How long is the program and when are classes held? | Curriculum, PDF p. 1 |
| C04 | Answerable | Which hyperparameter-tuning approaches are covered in Module 6? | Curriculum, PDF p. 2 |
| C05 | Answerable | What NLP concepts are listed in Module 8? | Curriculum, PDF p. 2 |
| C06 | Answerable | Which vector databases are named in Module 10? | Curriculum, PDF p. 2 |
| C07 | Answerable | What does Module 11 cover besides building agents? | Curriculum, PDF p. 3 |
| C08 | Answerable | What must the capstone integrate? | Curriculum, PDF p. 3 |
| C09 | Answerable | Which module introduces RAG and what is its outcome? | Curriculum, PDF p. 3 |
| C10 | Answerable | Which techniques are listed in Module 5? | Curriculum, PDF p. 2 |
| C11 | Answerable | Which neural architectures are named in Module 7? | Curriculum, PDF p. 2 |
| C12 | Answerable | What LLM topics are included in Module 9? | Curriculum, PDF p. 2 |
| B01 | Answerable | What do NumPy and SciPy each provide? | Book, PDF p. 27 / printed p. 6 |
| B02 | Answerable | How does the book describe classification? | Book, PDF p. 50 / printed p. 29 |
| B03 | Answerable | What flower classification problem appears in Chapter 2? | Book, PDF p. 50 / printed p. 29 |
| B04 | Answerable | Why is clustering described as unsupervised? | Book, PDF p. 72 / printed p. 51 |
| B05 | Answerable | How does topic modeling differ from clustering? | Book, PDF p. 100 / printed p. 79 |
| B06 | Answerable | What should the poor-answer classifier provide? | Book, PDF p. 116 / printed p. 95 |
| B07 | Answerable | Why is tweet sentiment analysis difficult? | Book, PDF p. 144 / printed p. 123 |
| B08 | Answerable | What regression problem opens Chapter 7? | Book, PDF p. 178 / printed p. 157 |
| B09 | Answerable | Why cannot song genres be handled like text? | Book, PDF p. 220 / printed p. 199 |
| B10 | Answerable | What image representation is used before classification? | Book, PDF p. 241 / printed p. 220 |
| B11 | Answerable | Give two reasons for dimensionality reduction. | Book, PDF p. 262 / printed p. 241 |
| B12 | Answerable | What tasks can `jug` perform? | Book, PDF p. 284 / printed p. 263 |
| N01 | Unanswerable | What is the curriculum tuition fee? | No evidence expected |
| N02 | Unanswerable | Who teaches Module 10? | No evidence expected |
| N03 | Unanswerable | What capstone grade is required to pass? | No evidence expected |
| N04 | Unanswerable | Which current scikit-learn version should the RAG project install? | No evidence expected |
| N05 | Unanswerable | What GPU should run Qwen 7B locally? | No evidence expected |
| N06 | Unanswerable | What is the newest LangChain release date? | No evidence expected |

## Expected output format for the application

```text
Answer: [grounded answer]

Sources: [file name — PDF page / table / section]
```

For an unanswerable case:

```text
Answer: I cannot find this information in the provided documents.

Best guess: No document evidence is sufficient to make a responsible inference.
```

## Notes for implementation

- Match **PDF page numbers** in the JSON to the physical 1-based page number in the source file. For the book, `printed_page` is also included as a human-friendly cross-reference.
- Do not use an external web search to answer or score these questions.
- Record the embedding model, chunking configuration, retrieval method, `top_k`, and LLM model for every evaluation run. This makes the similarity, reranker, and HyDE comparisons meaningful.
