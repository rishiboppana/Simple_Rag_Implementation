"""Core document-only RAG functions used by the Streamlit interface."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md", ".csv", ".ppt", ".pptx", ".xls", ".xlsx"}
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Settings:
    documents_dir: Path = Path(os.getenv("BASE_DOCUMENTS_DIR", "documents"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "data/chroma"))
    chat_model: str = os.getenv("HF_CHAT_MODEL", "Qwen/Qwen3-8B")
    embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))

    @property
    def manifest_path(self) -> Path:
        return self.chroma_dir.parent / "index_manifest.json"


class BGEEmbeddings(Embeddings):
    """Adds BGE's retrieval instruction to queries, but never to document chunks."""

    def __init__(self, model: str, token: str) -> None:
        self.client = HuggingFaceEndpointEmbeddings(model=model, huggingfacehub_api_token=token)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_query(f"{BGE_QUERY_INSTRUCTION}{text}")


def require_token() -> str:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN in .env before indexing or asking questions.")
    return token


def create_embeddings(settings: Settings) -> BGEEmbeddings:
    return BGEEmbeddings(settings.embedding_model, require_token())


def create_llm(settings: Settings) -> ChatHuggingFace:
    # the provider serving Qwen only supports the chat (conversational) task
    return ChatHuggingFace(llm=HuggingFaceEndpoint(
        repo_id=settings.chat_model,
        huggingfacehub_api_token=require_token(),
        task="conversational",
        max_new_tokens=512,
        temperature=0.1,
    ))


def discover_documents(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    elif suffix in {".txt", ".md"}:
        documents = TextLoader(str(path), encoding="utf-8").load()
    elif suffix == ".csv":
        documents = CSVLoader(str(path)).load()
    elif suffix == ".docx":
        documents = Docx2txtLoader(str(path)).load()
    else:
        # shortcut: office formats use Unstructured's generic parser; replace it only if a format needs specialized metadata.
        documents = UnstructuredFileLoader(str(path)).load()

    for document in documents:
        document.metadata.update({"source_file": path.name, "source_path": str(path), "file_type": suffix})
    return documents


def chunk_documents(documents: list[Document], embeddings: Embeddings) -> list[Document]:
    semantic_chunks = SemanticChunker(embeddings, breakpoint_threshold_type="percentile").split_documents(documents)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=120)
    final_chunks: list[Document] = []
    for semantic_index, semantic_chunk in enumerate(semantic_chunks):
        semantic_chunk.metadata["semantic_chunk_id"] = semantic_index
        final_chunks.extend(splitter.split_documents([semantic_chunk]))
    for chunk_index, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = chunk_index
    return final_chunks


def open_store(settings: Settings, embeddings: Embeddings, collection_name: str = "base_documents") -> Chroma:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(collection_name=collection_name, persist_directory=str(settings.chroma_dir), embedding_function=embeddings)


def read_manifest(settings: Settings) -> dict[str, str]:
    if not settings.manifest_path.exists():
        return {}
    return json.loads(settings.manifest_path.read_text())


def write_manifest(settings: Settings, manifest: dict[str, str]) -> None:
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def sync_base_documents(settings: Settings, embeddings: Embeddings, rebuild: bool = False) -> dict[str, int]:
    paths = discover_documents(settings.documents_dir)
    current = {str(path.resolve()): file_hash(path) for path in paths}
    previous = {} if rebuild else read_manifest(settings)
    store = open_store(settings, embeddings)

    if rebuild:
        try:
            store.delete_collection()
        except ValueError:
            pass
        store = open_store(settings, embeddings)

    removed = set(previous) - set(current)
    changed = [path for path in paths if previous.get(str(path.resolve())) != current[str(path.resolve())]]
    for source_path in removed:
        store.delete(where={"source_path": source_path})
    for path in changed:
        source_path = str(path.resolve())
        if source_path in previous:
            store.delete(where={"source_path": source_path})
        chunks = chunk_documents(load_file(path), embeddings)
        if chunks:
            ids = [f"{current[source_path]}-{chunk.metadata['chunk_id']}" for chunk in chunks]
            store.add_documents(chunks, ids=ids)

    write_manifest(settings, current)
    return {"indexed": len(changed), "removed": len(removed), "total": len(paths)}


def index_temporary_files(files: Iterable[tuple[str, bytes]], embeddings: Embeddings) -> Chroma:
    """Create an in-memory Chroma collection for one Streamlit chat session."""
    import tempfile

    # Streamlit keeps the returned store alive for the active session; its temporary directory is intentionally process-scoped.
    directory = tempfile.mkdtemp(prefix="rag-temp-")
    documents: list[Document] = []
    for filename, content in files:
        path = Path(directory) / filename
        path.write_bytes(content)
        documents.extend(load_file(path))
    chunks = chunk_documents(documents, embeddings)
    return Chroma.from_documents(chunks, embedding=embeddings, collection_name=f"temporary_{hashlib.sha1(directory.encode()).hexdigest()[:10]}")


def retrieve(question: str, stores: Iterable[Chroma], k: int) -> list[tuple[Document, float]]:
    results: list[tuple[Document, float]] = []
    for store in stores:
        results.extend(store.similarity_search_with_relevance_scores(question, k=k))
    return sorted(results, key=lambda item: item[1], reverse=True)[:k]


def format_context(results: list[tuple[Document, float]]) -> str:
    return "\n\n".join(
        f"[SOURCE {index + 1}: {document.metadata.get('source_file', 'unknown')}, page {document.metadata.get('page', 'n/a')}]\n{document.page_content}"
        for index, (document, _) in enumerate(results)
    )


def answer_question(question: str, history: list[dict[str, str]], results: list[tuple[Document, float]], settings: Settings) -> str:
    if not results or results[0][1] < settings.min_relevance_score:
        return "I cannot find this information in the provided documents.\n\nBest guess: No document evidence is sufficient to make a responsible inference."

    recent_history = history[-4:]
    conversation = "\n".join(f"User: {item['question']}\nAssistant: {item['answer']}" for item in recent_history)
    prompt = f"""You are a document-only RAG assistant. Answer only from the retrieved source context. Do not use outside knowledge.
If the context is insufficient, say: I cannot find this information in the provided documents.
If partial evidence supports an inference, place it on a new line beginning with 'Best guess:' and label it as an inference.
Be concise and do not invent citations; the application appends sources.

Conversation history:
{conversation or '(none)'}

Retrieved source context:
{format_context(results)}

Question: {question}
Answer:"""
    # Qwen3 may emit a <think> block before the answer
    return re.sub(r"<think>.*?</think>", "", create_llm(settings).invoke(prompt).content, flags=re.DOTALL).strip()


def source_labels(results: list[tuple[Document, float]]) -> str:
    labels: list[str] = []
    for document, _ in results:
        source = document.metadata.get("source_file", "unknown")
        page = document.metadata.get("page")
        label = f"{source} — page {int(page) + 1}" if isinstance(page, int) else source
        if label not in labels:
            labels.append(label)
    return "; ".join(labels)
