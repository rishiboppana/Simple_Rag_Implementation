# Document RAG

Streamlit RAG application for answering questions from local documents only.

## Chosen approach

- **UI:** Streamlit
- **Base documents:** files inside `documents/`, synchronized incrementally with hashes
- **Temporary uploads:** indexed for the current Streamlit chat only
- **Chunking:** semantic chunks, then 800-token fixed-size sub-chunks with 120-token overlap
- **Embeddings:** `BAAI/bge-base-en-v1.5` through Hugging Face Inference
- **Vector store:** persistent local ChromaDB
- **Generation:** configurable Qwen 7B instruction model through Hugging Face Inference
- **Retrieval:** similarity search, top 4 chunks, across the whole collection
- **Grounding:** document-only answers with source file/page labels and strict abstention

## Run locally

1. Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add `HUGGINGFACEHUB_API_TOKEN`.
3. Put the source files in `documents/`.
4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Select **Sync document folder** in the sidebar.

## Evaluation

Use [`outputs/rag_evaluation_set_30.json`](outputs/rag_evaluation_set_30.json) and [`outputs/rag_evaluation_guide.md`](outputs/rag_evaluation_guide.md) to compare the similarity baseline, reranking, and HyDE experiments.

## Current boundary

Only files containing extractable text are in scope. Image-only/scanned PDFs require OCR and are intentionally not processed in this version.
