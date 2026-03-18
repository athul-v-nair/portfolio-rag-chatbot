# Portfolio RAG Chatbot

A retrieval-augmented generation (RAG) pipeline that ingests personal documents and serves them as a conversational chatbot for a portfolio page.

---

## Overview

The system processes raw documents (PDF, Markdown, plain text) through a structured ingestion pipeline, stores vector embeddings in ChromaDB, and exposes a retrieval interface for a generation layer to answer questions grounded in the ingested content.

The core challenge with resume PDFs is that standard document parsers (including `unstructured`) split content at arbitrary visual boundaries — titles land in one chunk, their content in another. This makes retrieved context incoherent and degrades answer quality.
 
This pipeline solves that by using a Gemini LLM to read the full document and return a structured JSON object with canonical resume sections. That structured output is converted back into LangChain Documents with clean section metadata, chunked, embedded, and stored in ChromaDB.

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Load      │───>│   Parse          │───>│   Chunk     │───>│   Embed     │───>│   Store     │
│             │    │                  │    │             │    │             │    │             │
│ PDF / MD /  │    │ Send to          │    │ Section-    │    │ Google      │    │ ChromaDB    │
│ TXT via     │    │ Gemini with      │    │ aware for   │    │ Gemini      │    │ (persisted  │
│ LangChain   │    │ structured       │    │ PDF         │    │ Embeddings  │    │ on disk)    │
│ loaders     │    │ prompt,returns   │    │             │    │ w/ HF       │    │             │
│             │    │ JSON. Convert    │    │             │    │ fallback    │    │             │
│             │    │ to Documents     │    │             │    │             │    │             │
└─────────────┘    └──────────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Load:** LangChain loaders (`PyPDFLoader`, `UnstructuredMarkdownLoader`, `TextLoader`) read files from `data/raw` and attach `file_type` and `file_name` metadata.

**Parse:** Pages are first reassembled into a single full-text string per file, since section content frequently spans page breaks. The full text is sent to `gemini-2.5-flash` with a structured prompt that instructs the model to return a JSON object with keys: `summary`, `skills`, `projects`, `experience`, `education`. The JSON response is parsed and each section is converted into a LangChain Document with a `section` metadata key.

**Chunk:** A section-aware chunker groups parsed elements by `(file_name, section)`. If a section fits within the configured `chunk_size`, it becomes a single chunk. Oversized sections are further split with `RecursiveCharacterTextSplitter` while preserving the `section` metadata on every resulting chunk. Plain text paragraphs follow a simpler path — each paragraph is a chunk unless it exceeds the size limit.

**Embed:** Chunks are encoded using `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2-preview`). If the Google API is unavailable, the pipeline falls back to `HuggingFaceEndpointEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`).

**Store:** Embeddings and metadata are persisted in a local ChromaDB collection. The store supports incremental upserts and full rebuilds.

---
 
## Why LLM-based Parsing
 
Standard layout parsers (`unstructured`, `pdfplumber`) detect visual structure. Resumes are not consistently formatted — a heading may be a bold inline span, a coloured block, or just an all-caps word. These parsers reliably separate titles from their content, producing chunks with a heading and no body, or a body and no heading.
 
Sending the full text to an LLM sidesteps layout detection entirely. The model understands semantic structure regardless of visual formatting, and the prompt constrains its output to a schema that maps directly to the Documents needed downstream.
 
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
    │   ├── document_parser.py  # Page reassembly, LLM sectioning, JSON-to-Document conversion
    │   ├── chunker.py          # Chunk ID assignment; splits oversized sections
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

# Chunk Metadata
 
Every stored chunk carries the following metadata, available for filtered retrieval:
 
| Key           | Example value                        | Description                          |
|---------------|-----------------------------------------|--------------------------------------|
| `file_name`   | `Athul_AI_Engineer_Resume.pdf`          | Source file                          |
| `section`     | `project`                               | Resume section identified by the LLM |
| `chunk_index` | `0`                                     | Position within the section          |
| `chunk_id`    | `Athul_resume_section_projects_3f2a1b4c`| Unique ID for the chunk in ChromaDB  |
| `college`     | `MIT` *(education only)*                | College name for filtered retrieval  |
 
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