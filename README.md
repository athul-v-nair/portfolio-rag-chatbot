# Portfolio RAG Chatbot

A retrieval-augmented generation (RAG) pipeline that ingests personal documents and serves them as a conversational chatbot for a portfolio page.

---

## Overview

The system processes raw documents (PDF, Markdown, plain text) through a structured ingestion pipeline, stores vector embeddings in ChromaDB, and exposes a retrieval interface for a generation layer to answer questions grounded in the ingested content.

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Load      │───>│   Parse     │───>│   Chunk     │───>│   Embed     │───>│   Store     │
│             │    │             │    │             │    │             │    │             │
│ PDF / MD /  │    │ Unstructured│    │ Section-    │    │ Google      │    │ ChromaDB    │
│ TXT via     │    │ partition   │    │ aware for   │    │ Gemini      │    │ (persisted  │
│ LangChain   │    │ + paragraph │    │ PDF / MD    │    │ Embeddings  │    │ on disk)    │
│ loaders     │    │ splitting   │    │ Paragraph   │    │ w/ HF       │    │             │
│             │    │             │    │ for TXT     │    │ fallback    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Load:** LangChain loaders (`PyPDFLoader`, `UnstructuredMarkdownLoader`, `TextLoader`) read files from `data/raw` and attach `file_type` and `file_name` metadata.

**Parse:** Documents are parsed by type. PDFs and Markdown files are processed with `unstructured` (`partition_pdf`, `partition_md`), which identifies structural elements such as titles, narrative text, and list items. Section context is tracked as elements are iterated. Plain text files are split on double newlines into paragraphs.

**Chunk:** A section-aware chunker groups parsed elements by `(file_name, section)`. If a section fits within the configured `chunk_size`, it becomes a single chunk. Oversized sections are further split with `RecursiveCharacterTextSplitter` while preserving the `section` metadata on every resulting chunk. Plain text paragraphs follow a simpler path — each paragraph is a chunk unless it exceeds the size limit.

**Embed:** Chunks are encoded using `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2-preview`). If the Google API is unavailable, the pipeline falls back to `HuggingFaceEndpointEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`).

**Store:** Embeddings and metadata are persisted in a local ChromaDB collection. The store supports incremental upserts and full rebuilds.

---

## Project Structure

```
.
├── data/
│   ├── raw/                    # Source documents (PDF, MD, TXT)
│   └── db/                     # ChromaDB persisted files
│
└── src/
    ├── ingestion/
    │   ├── loaders.py          # File type detection and LangChain loaders
    │   ├── document_parser.py  # Unstructured parsing, section tracking
    │   ├── chunker.py          # Section-aware and paragraph chunking
    │   ├── embedding.py        # Embedding model init with fallback logic
    │   ├── vector_store.py     # ChromaDB wrapper (add, search, retriever)
    │   └── pipeline.py         # Orchestrates all ingestion stages
    │
    ├── generation/             # (in progress) LLM chain and prompt logic
    ├── retrival/               # (in progress) Retriever configurations
    │
    └── utils/
        ├── constants.py        # Paths, model names, config values
        └── logger.py           # Shared logger setup
```

---

## Setup

**Prerequisites:** Python 3.10+

```bash
pip install -r requirements.txt
```

**Environment variables:**

```env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key   # fallback only
```

**Add source documents:**

Place `.pdf`, `.md`, or `.txt` files in `data/raw/`.

---

## Running the Pipeline

```bash
# Ingest documents
python src/ingestion/pipeline.py
```

---

## Key Design Decisions

**Section-aware chunking over naive splitting.** Splitting on a fixed character window ignores document structure. By grouping elements under their detected section heading before chunking, retrieved context is semantically coherent and the LLM receives complete thoughts rather than fragments spanning two unrelated sections.

**Metadata preserved end-to-end.** Every chunk retains `file_name`, `section`, `element_type`, `chunk_type`, and (for PDFs) `page_number`. This allows the generation layer to cite sources precisely and supports metadata-filtered retrieval — for example, restricting search to a specific document or section.

**Embedding fallback.** Gemini embeddings offer high quality but require a Google API key and network access. The HuggingFace fallback keeps the pipeline functional in offline or cost-sensitive environments with no code changes required.

**Incremental upserts.** Running the pipeline on a partially-updated `data/raw` directory adds new chunks without touching existing ones. Use `--rebuild` only when a full re-index is needed.

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-community` | Loaders, splitters, vector store abstraction |
| `langchain-google-genai` | Gemini embedding model |
| `langchain-huggingface` | HuggingFace fallback embeddings |
| `unstructured[pdf,md]` | Structural parsing of PDF and Markdown |
| `chromadb` | Local vector database |
| `pypdf` | PDF loading |

## Author

**Athul V Nair** — [GitHub](https://github.com/athul-v-nair)

*Build for integrating with Personal Portfolio Page*