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
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Load    │──▶│  Parse   │──▶│  Chunk   │──▶│  Embed   │──▶│  Store   │
│          │   │          │   │          │   │          │   │          │
│ PDF/MD/  │   │ Gemini   │   │ Section- │   │ Gemini   │   │ ChromaDB │
│ TXT via  │   │ 2.5 Flash│   │ aware    │   │ Embed /  │   │ persisted│
│ LangChain│   │ → JSON   │   │ chunker  │   │ HF backup│   │ on disk  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
 
                              ┌──────────────────────────────────────┐
                              │             Query Flow                │
                              │                                       │
                              │  User query ──▶ ChromaDB retrieval   │
                              │       ──▶ Gemini 2.5 Flash generation │
                              │       ──▶ Grounded answer + sources  │
                              └──────────────────────────────────────┘
```

**Load:** LangChain loaders (`PyPDFLoader`, `UnstructuredMarkdownLoader`, `TextLoader`) read files from `data/raw` and attach `file_type` and `file_name` metadata.

**Parse:** Pages are first reassembled into a single full-text string per file, since section content frequently spans page breaks. The full text is sent to `gemini-2.5-flash` with a structured prompt that instructs the model to return a JSON object with keys: `summary`, `skills`, `projects`, `experience`, `education`. The JSON response is parsed and each section is converted into a LangChain Document with a `section` metadata key.

**Chunk:** A section-aware chunker groups parsed elements by `(file_name, section)`. If a section fits within the configured `chunk_size`, it becomes a single chunk. Oversized sections are further split with `RecursiveCharacterTextSplitter` while preserving the `section` metadata on every resulting chunk. Plain text paragraphs follow a simpler path — each paragraph is a chunk unless it exceeds the size limit.

**Embed:** Chunks are encoded using `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2-preview`). If the Google API is unavailable, the pipeline falls back to `HuggingFaceEndpointEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`).

**Store:** Embeddings and metadata are persisted in a local ChromaDB collection. The store supports incremental upserts and full rebuilds.

**Retrieve:** `vector_search.py` runs a similarity search against ChromaDB, returning the top-K chunks sorted by L2 distance (lower = more similar).
 
**Generate:** `generation.py` filters chunks by score threshold, builds a grounded prompt (resume context + conversation history + user query), and calls Gemini 2.5 Flash. Answers are strictly grounded in retrieved context — the model is instructed never to fabricate.

---
 
## Why LLM-based Parsing
 
Standard layout parsers (`unstructured`, `pdfplumber`) detect visual structure. Resumes are not consistently formatted — a heading may be a bold inline span, a coloured block, or just an all-caps word. These parsers reliably separate titles from their content, producing chunks with a heading and no body, or a body and no heading.
 
Sending the full text to an LLM sidesteps layout detection entirely. The model understands semantic structure regardless of visual formatting, and the prompt constrains its output to a schema that maps directly to the Documents needed downstream.
 
---

## Project Structure

```
.
├── data/
│   ├── raw/                        # Source documents (PDF, MD, TXT)
│   └── db/                         # ChromaDB persisted files
│
├── api/
│   └── api.py                      # FastAPI app — POST /chat, GET /health
│
└── src/
    ├── ingestion/
    │   ├── loaders.py              # File type detection and LangChain loaders
    │   ├── document_parser.py      # Page reassembly, LLM sectioning, JSON → Document
    │   ├── chunker.py              # Chunk ID assignment; splits oversized sections
    │   ├── embedding.py            # Embedding model init with HF fallback
    │   ├── vector_store.py         # ChromaDB wrapper (add, upsert, search)
    │   └── pipeline.py             # Orchestrates all ingestion stages
    │
    ├── retrieval/
    │   └── vector_search.py        # Similarity search; returns scored chunks
    │
    ├── generation/
    │   ├── generation.py           # RAGGenerator: filter → prompt → Gemini → result
    │   └── memory.py               # Memory store for maintaining recent chat history.
    │
    └── utils/
        ├── prompts/
        │   ├── generation_prompt.py  # System prompt + context + history assembly
        │   └── parsing_prompt.py     # Structured JSON extraction prompt for resume
        ├── constants.py              # Paths, model names, config values
        └── logger.py                 # Shared logger setup
```

---

## Setup

**Prerequisites:** Python 3.10+

```bash
pip install -r requirements.txt
```

**Environment variables** — copy the block below into a `.env` file at the project root:
 
```env
GEMINI_API_KEY='your_google_api_key'
HUGGINGFACE_API_KEY='your_huggingface_api_key'          # fallback only
 
# Vector DB
COLLECTION_NAME="portfolio_rag"
DB_PATH='data/db'
 
# Document paths
DOCS_PATH="data/raw"
 
# Embedding models
GEMINI_EMBEDDING_MODEL="gemini-embedding-2-preview"
BACKUP_EMBEDDING_MODEL="all-MiniLM-L6-v2"
BACKUP_EMBEDDING_MODEL_REPO_ID="sentence-transformers/all-MiniLM-L6-v2"
 
# Text generation
GEMINI_TEXT_GENERATION_MODEL="gemini-2.5-flash"
 
# Generation tuning
GENERATION_TEMPERATURE="0.2"
GENERATION_MAX_TOKENS="512"
RETRIEVAL_TOP_K="3"
SCORE_THRESHOLD="0.45"

# Chat history
NUMBER_OF_CHATS="10"
```

**Add source documents:**

Place `.pdf`, `.md`, or `.txt` files in `data/raw/`.

---

## Running the Project
 
### 1. Ingest documents
 
```bash
python src/ingestion/pipeline.py
```
 
For a full rebuild (drops and recreates the ChromaDB collection):
 
```bash
python src/ingestion/pipeline.py --rebuild
```
 
### 2. Start the API server
 
```bash
uvicorn api.api:app --reload --port 8000
```
 
The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.
 
### 3. Send a query
 
**Simple query:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What projects has Athul built?", "session_id": "uuid_0001_x3d"}'
```

**Example response:**
```json
{
  "answer": "Athul built Immersify using AWS Bedrock, AWS Transcribe, and AWS S3.",
  "sources": [
    {"section": "projects", "file_name": "Athul_AI_Engineer_Resume.pdf", "chunk_id": "..."}
  ],
  "latency_ms": 1243,
  "was_context_used": true
}
```

---

## Key Design Decisions

**Section-aware chunking over naive splitting.** Splitting on a fixed character window ignores document structure. By grouping elements under their detected section heading before chunking, retrieved context is semantically coherent and the LLM receives complete thoughts rather than fragments spanning two unrelated sections.

**Metadata preserved end-to-end.** Every chunk retains `file_name`, `section`, `element_type`, `chunk_type`, and (for PDFs) `page_number`. This allows the generation layer to cite sources precisely and supports metadata-filtered retrieval — for example, restricting search to a specific document or section.

**Embedding fallback.** Gemini embeddings offer high quality but require a Google API key and network access. The HuggingFace fallback keeps the pipeline functional in offline or cost-sensitive environments with no code changes required.

**Incremental upserts.** Running the pipeline on a partially-updated `data/raw` directory adds new chunks without touching existing ones. Use `--rebuild` only when a full re-index is needed.

**Stateless chat history.** Server-side session storage adds infrastructure complexity and breaks on free-tier stateless hosting. Keeping history on the client is simpler, more resilient, and equally effective for 5–10 turn conversations.

---

## Chat History
 
The API is **stateless** — the server stores no session data. The frontend is responsible for maintaining history and sending the last N turns with each request.
 
This design works with zero infrastructure on any free-tier hosting platform. The server enforces a hard cap of 10 messages; if the frontend sends more, older turns are silently trimmed.
 
**Recommended frontend pattern:**
 
```javascript
const history = [];
 
async function sendMessage(userQuery) {
  const response = await fetch('/chat', {
    method: 'POST',
    body: JSON.stringify({ query: userQuery, history: history.slice(-10) })
  });
  const data = await response.json();
 
  // Append both turns to local history
  history.push({ role: 'user',      content: userQuery });
  history.push({ role: 'assistant', content: data.answer });
 
  return data.answer;
}
```

---

# Chunk Metadata
 
Every stored chunk carries the following metadata, available for filtered retrieval:
 
| Key           | Example value                           | Description                          |
|---------------|-----------------------------------------|--------------------------------------|
| `file_name`   | `Athul_AI_Engineer_Resume.pdf`          | Source file                          |
| `section`     | `project`                               | Resume section identified by the LLM |
| `chunk_index` | `0`                                     | Position within the section          |
| `chunk_id`    | `Athul_resume_section_projects_3f2a1b4c`| Unique ID for the chunk in ChromaDB  |
| `college`     | `MIT` *(education only)*                | College name for filtered retrieval  |


---
 
## Roadmap
 
- [x] Data ingestion pipeline (PDF → structured sections → ChromaDB)
- [x] Vector retrieval (`similarity_search_with_score`, top-K)
- [x] Generation layer (Gemini 2.5 Flash, score-filtered context)
- [x] FastAPI serving (`/chat`, `/health`, CORS)
- [ ] Conversation history (stateless, client-owned, 10-turn cap)
- [ ] RAG evaluation (faithfulness, answer relevance, context recall)
- [ ] Markdown and plain text ingestion
- [ ] Reranking (if retrieval precision degrades at scale)
 
---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-community` | Loaders, splitters, vector store abstraction |
| `langchain-google-genai` | Gemini embedding model, Gemini Text Generation Model |
| `langchain-huggingface` | HuggingFace fallback embeddings |
| `chromadb` | Local vector database |
| `pypdf` | PDF loading |

## Author

**Athul V Nair** — [GitHub](https://github.com/athul-v-nair)

<p align="center"><i>Build for integrating with Personal Portfolio Page</i></p>
