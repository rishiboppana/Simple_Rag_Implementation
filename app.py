from __future__ import annotations

import streamlit as st
from sentence_transformers import CrossEncoder

from rag import Settings, answer_question, create_embeddings, discover_documents, index_temporary_files, open_store, retrieve, source_labels, sync_base_documents

st.set_page_config(page_title="Document RAG", page_icon="📚", layout="wide")
settings = Settings(min_relevance_score=0.0)  # no cutoff: the LLM alone decides to abstain
RERANK_K, FETCH = 8, 20


@st.cache_resource(show_spinner=False)
def embeddings():
    return create_embeddings(settings)


@st.cache_resource(show_spinner=False)
def reranker():
    return CrossEncoder("BAAI/bge-reranker-base")


def rerank(question, stores):
    candidates = retrieve(question, stores, FETCH)
    scores = reranker().predict([(question, doc.page_content) for doc, _ in candidates])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:RERANK_K]
    return [candidates[i] for i in order]


def ensure_history() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("temporary_store", None)


ensure_history()
st.title("Document RAG")
st.caption("Answers are generated only from your indexed documents.")

with st.sidebar:
    st.subheader("Base documents")
    st.code(str(settings.documents_dir.resolve()), language=None)
    st.write(f"{len(discover_documents(settings.documents_dir))} supported file(s) found")
    if st.button("Sync document folder", use_container_width=True):
        with st.spinner("Checking and indexing changed documents..."):
            result = sync_base_documents(settings, embeddings())
        st.success(f"Indexed {result['indexed']}, removed {result['removed']}; {result['total']} file(s) in the folder.")
    if st.button("Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding the full index..."):
            result = sync_base_documents(settings, embeddings(), rebuild=True)
        st.success(f"Rebuilt the index from {result['total']} file(s).")

    st.subheader("Temporary documents")
    uploads = st.file_uploader("Add files to this chat only", accept_multiple_files=True)
    if uploads and st.button("Index temporary uploads", use_container_width=True):
        with st.spinner("Indexing temporary files..."):
            st.session_state.temporary_store = index_temporary_files([(file.name, file.getvalue()) for file in uploads], embeddings())
        st.success(f"Added {len(uploads)} temporary file(s) for this chat.")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        st.caption(f"Sources: {item['sources']}")

question = st.chat_input("Ask a question about the documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            stores = [open_store(settings, embeddings())]
            if st.session_state.temporary_store is not None:
                stores.append(st.session_state.temporary_store)
            with st.spinner("Retrieving document evidence..."):
                results = rerank(question, stores)
                answer = answer_question(question, st.session_state.history, results, settings)
            sources = source_labels(results)
            st.write(answer)
            if sources:
                st.caption(f"Sources: {sources}")
            st.session_state.history.append({"question": question, "answer": answer, "sources": sources})
        except Exception as error:
            st.error(f"Unable to answer this question: {error}")
