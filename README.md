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
│ TXT via  │   │ 2.5 Flash│   │ aware    │   │ Embedding│   │ persisted│
│ LangChain│   │ → JSON   │   │ chunker  │   │          │   │ on disk  │
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

**Embed:** Chunks are encoded using `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2-preview`). If the Google API is unavailable, embedding fails and raises an error. The fallback to Hugging Face embeddings has been removed, since any mismatch between the vector embedding model and the query embedding model would cause retrieval to fail.

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
│   ├── raw/                          # Source documents (PDF, MD, TXT)
│   └── db/                           # ChromaDB persisted files
│
├── api/
│   └── api.py                        # FastAPI app — POST /chat, GET /health
│
└── src/
    ├── ingestion/
    │   ├── loaders.py                # File type detection and LangChain loaders
    │   ├── document_parser.py        # Page reassembly, LLM sectioning, JSON → Document
    │   ├── chunker.py                # Chunk ID assignment; splits oversized sections
    │   ├── embedding.py              # Gemini embedding model init
    │   ├── vector_store.py           # ChromaDB wrapper (add, upsert, search)
    │   └── pipeline.py               # Orchestrates all ingestion stages
    │
    ├── retrieval/
    │   └── vector_search.py          # Similarity search; returns scored chunks
    │
    ├── generation/
    │   ├── generation.py             # RAGGenerator: filter → prompt → Gemini → result
    │   └── memory.py                 # In-memory store for recent chat history
    │
    ├── evaluation/
    │   ├── evaluator.py              # RAGEvaluator: retrieval + generation metrics
    │   ├── evaluation_pipeline.py    # Runs evaluation and prints summary
    │   └── evaluation_dataset.json   # Ground-truth Q&A pairs with keyword anchors
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
 
# Vector DB
COLLECTION_NAME="portfolio_rag"
DB_PATH='data/db'
 
# Document paths
DOCS_PATH="data/raw"
 
# Embedding models
GEMINI_EMBEDDING_MODEL="gemini-embedding-2-preview"
 
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

### 4. Run evaluation
 
```bash
python src/evaluation/evaluation_pipeline.py
```

---

## API Reference
 
### `POST /chat`
 
| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | User's question (max 500 chars) |
| `session_id` | string | Yes | Session identifier for chat history lookup |
| `top_k` | int | No | Chunks to retrieve (default: 3, max: 10) |

---
 
## Chat History
 
Chat history is maintained server-side in memory, keyed by `session_id`. The server retains the last `NUMBER_OF_CHATS` turns (default: 10) per session and injects them into the generation prompt so the model can resolve follow-up questions like "which of those used AWS?".
 
History is scoped to the current process and is not persisted across server restarts. For a portfolio chatbot this is an acceptable trade-off — sessions are short-lived and free-tier hosting platforms restart infrequently.
 
---
 
## RAG Evaluation
 
The evaluation module measures both retrieval quality and generation quality against a curated ground-truth dataset.
 
### Metrics
 
| Metric | What it measures |
|---|---|
| **Recall@K** | Whether at least one relevant chunk was retrieved in the top-K results |
| **MRR** (Mean Reciprocal Rank) | How highly the first relevant chunk is ranked — 1.0 means it was always rank 1 |
| **Context Precision** | Fraction of retrieved chunks that are actually relevant — penalizes noisy retrieval |
| **Answer Similarity** | Token-level overlap between the generated answer and the ground-truth answer |
 
### Running the evaluator
 
```bash
python src/evaluation/evaluation_pipeline.py
```
 
The dataset lives at `src/evaluation/evaluation_dataset.json`. Each entry contains a query, a ground-truth answer, and keyword anchors used to judge chunk relevance:
 
```json
{
  "query": "Which AWS services were used in Immersify?",
  "ground_truth_answer": "AWS Bedrock, AWS Transcribe, AWS S3, AWS OpenSearch",
  "relevant_doc_keywords": ["Bedrock", "Transcribe", "S3", "OpenSearch"]
}
```
 
### Results (v1 — 6 queries, top_k=3)
 
| Query | Recall@3 | MRR | Context Precision | Answer Similarity |
|---|---|---|---|---|
| What projects has Athul built? | 1.0 | 1.0 | 0.67 | 0.50 |
| What is Immersify? | 1.0 | 1.0 | 0.33 | 0.40 |
| Which AWS services were used in Immersify? | 1.0 | 1.0 | 0.33 | 0.40 |
| What model architecture was used in the transformer project? | 1.0 | 1.0 | 0.67 | 0.60 |
| What technologies are used in Contrack? | 1.0 | 1.0 | 0.67 | 0.25 |
| What does the perceptron project demonstrate? | 1.0 | 1.0 | 0.33 | 0.60 |
| **Average** | **1.0** | **1.0** | **0.50** | **0.46** |
 
### Interpretation
 
**Retrieval is strong.** Perfect Recall@3 and MRR of 1.0 across all queries means the correct chunk is always retrieved and always ranked first. No reranking is needed at this scale.
 
**Context Precision of 0.50** means on average one of the three retrieved chunks is not directly relevant. This is expected with top_k=3 on a small document — reducing to top_k=2 is worth experimenting with given the perfect MRR already places the correct chunk at rank 1.
 
**Answer Similarity of 0.46** reflects the limitations of token-overlap scoring rather than a generation quality problem. The metric counts exact word matches against a short reference string; a correct but more verbose answer scores lower. Qualitative review confirms the answers are factually accurate. Replacing this metric with an embedding-based semantic similarity score is the next evaluation improvement.

---

## Key Design Decisions

**Section-aware chunking over naive splitting.** Splitting on a fixed character window ignores document structure. By grouping elements under their detected section heading before chunking, retrieved context is semantically coherent and the LLM receives complete thoughts rather than fragments spanning two unrelated sections.

**Metadata preserved end-to-end.** Every chunk retains `file_name`, `section`, `element_type`, `chunk_type`, and (for PDFs) `page_number`. This allows the generation layer to cite sources precisely and supports metadata-filtered retrieval — for example, restricting search to a specific document or section.

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
- [x] Conversation history (stateless, client-owned, 10-turn cap)
- [x] RAG evaluation (Recall@K, MRR, Context Precision, Answer Similarity)
- [x] Docker setup and containerized deployment
- [ ] Rate Limiting and Generation Model Fallback
- [ ] Semantic answer similarity metric (embedding-based, replaces token overlap)
- [ ] Markdown and plain text ingestion
- [ ] Reranking (if retrieval precision degrades at scale)
 
---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain`, `langchain-community` | Loaders, splitters, vector store abstraction |
| `langchain-google-genai` | Gemini embedding model, Gemini Text Generation Model |
| `chromadb` | Local vector database |
| `pypdf` | PDF loading |
| `fastapi`, `uvicorn` | API server |
| `pydantic` | Request/response validation |
| `python-dotenv` | Environment variable loading |
| `numpy` | Evaluation metric aggregation |

## Author

**Athul V Nair** — [GitHub](https://github.com/athul-v-nair)

<p align="center"><i>Build for integrating with Personal Portfolio Page</i></p>
