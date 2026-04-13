# DocSage - Document Question Answering System

DocSage is an end-to-end document QA application with:

- a `FastAPI` backend for ingestion, retrieval, and answering
- a `React + Vite` frontend for upload, chat, and document filtering
- a hybrid retrieval stack (dense + sparse + graph signal)
- source-cited answers with confidence and risk metadata

The code for the app lives under `docsage/`.

## What It Does

1. Upload documents (`PDF`, `DOCX`, `TXT`, `MD`).
2. Parse and chunk them for retrieval.
3. Build/update hybrid indices.
4. Ask natural-language questions over all docs or selected docs.
5. Return answer + confidence + source evidence.

## High-Level Architecture

- **Frontend** (`docsage/frontend`): React UI for document and chat workflows.
- **Backend API** (`docsage/backend/api`): REST endpoints for documents, sessions, QA.
- **Pipeline Orchestration** (`docsage/backend/core/pipeline.py`): ingestion + QA flow.
- **Retrieval Stack** (`docsage/backend/utils/retriever.py`):
  - Dense retrieval (`SentenceTransformer` + `FAISS`)
  - Sparse retrieval (`BM25`)
  - Graph signal (`NetworkX`)
- **Answer Generation**:
  - reranking (`docsage/backend/models/reranker.py`)
  - extractive QA reader (`docsage/backend/models/reader.py`)

## Repository Structure

```text
.
├── README.md
└── docsage/
    ├── backend/
    │   ├── api/
    │   │   ├── main.py
    │   │   └── routers/
    │   │       ├── documents.py
    │   │       ├── qa.py
    │   │       └── sessions.py
    │   ├── core/
    │   │   ├── config.py
    │   │   ├── logging.py
    │   │   └── pipeline.py
    │   ├── models/
    │   │   ├── reader.py
    │   │   └── reranker.py
    │   ├── utils/
    │   │   ├── chunker.py
    │   │   ├── document_parser.py
    │   │   └── retriever.py
    │   ├── tests/
    │   │   ├── test_api.py
    │   │   └── test_chunker.py
    │   ├── requirements.txt
    │   └── data/                 # runtime artifacts (docs, indices, model cache)
    ├── frontend/
    │   ├── package.json
    │   └── src/
    │       ├── App.jsx
    │       ├── components/
    │       ├── services/api.js
    │       └── styles/
    ├── docker/
    │   ├── docker-compose.yml
    │   ├── Docker.backend
    │   └── Docker.frontend
    ├── scripts/
    │   ├── evaluate.py
    │   └── fine_tune.py
    └── configs/
```

## Core Runtime Flow

### Ingestion Pipeline

1. `POST /api/v1/documents/upload`
2. Save uploaded file in `backend/data/documents`
3. Background ingestion task:
   - parse sections (`document_parser.py`)
   - chunk text (`chunker.py`)
   - add chunks to hybrid retriever (`retriever.py`)
4. Persist index artifacts in `backend/data/indices`

### QA Pipeline

1. `POST /api/v1/qa/ask`
2. Optional adversarial-risk scoring of the query
3. Hybrid retrieval candidates (dense + sparse + graph)
4. Cross-encoder reranking
5. Extractive reader produces answer + confidence
6. API responds with answer, confidence label, latency, and source snippets

## Tech Stack

### Backend

- API: `fastapi`, `uvicorn`, `python-multipart`
- ML/NLP: `torch`, `transformers`, `sentence-transformers`
- Retrieval: `faiss-cpu`, `rank-bm25`, `networkx`
- Parsing: `pdfplumber`, `PyPDF2`, `python-docx`, `pytesseract`
- Config: `pydantic`, `pydantic-settings`

### Frontend

- `react`, `react-dom`
- `vite`, `@vitejs/plugin-react`

## Local Development Setup

## Prerequisites

- Python `3.11+` recommended
- Node `18+` (Node `20` recommended)
- `pip` and `npm`
- Optional for OCR-heavy PDFs: `tesseract` installed on host

## 1) Run Backend

```bash
cd docsage/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs:

- Swagger: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`
- Health: `http://localhost:8000/api/v1/health`

## 2) Run Frontend

```bash
cd docsage/frontend
npm install
npm run dev
```

Default frontend URL is usually `http://localhost:5173`.

If needed, set API base:

```bash
export VITE_API_URL="http://localhost:8000/api/v1"
```

## Docker Setup

Compose file: `docsage/docker/docker-compose.yml`

```bash
cd docsage/docker
docker compose up --build
```

### Note

Current compose references `Dockerfile.backend` / `Dockerfile.frontend`, while repository files are named `Docker.backend` / `Docker.frontend`. Update either compose paths or file names before using Docker end-to-end.

## API Endpoints

Base prefix: `/api/v1`

- `GET /health` - liveness/version
- `POST /documents/upload` - upload and queue ingestion
- `GET /documents/` - list documents
- `GET /documents/{doc_id}` - ingestion/status details
- `DELETE /documents/{doc_id}` - delete doc and remove index entries
- `POST /sessions/` - create chat session
- `GET /sessions/{session_id}/history` - get conversation history
- `DELETE /sessions/{session_id}/history` - clear history
- `POST /qa/ask` - ask question with optional document filter

### Example QA Request

```bash
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings?",
    "session_id": "your-session-id",
    "doc_ids": null
  }'
```

## Configuration

Backend settings are defined in `docsage/backend/core/config.py` and can be overridden via environment variables.

Common settings:

- `ENVIRONMENT` (`development|staging|production`)
- `API_PREFIX` (default `/api/v1`)
- `CORS_ORIGINS`
- `MAX_UPLOAD_SIZE_MB`
- `READER_MODEL_NAME`
- `EMBEDDER_MODEL_NAME`
- `RERANKER_MODEL_NAME`
- `ROBUSTNESS_MODEL_NAME`
- `DATABASE_URL`
- `REDIS_URL`

## Testing

From `docsage/backend`:

```bash
pytest -q
```

Current tests include:

- API endpoint tests (`tests/test_api.py`)
- chunker logic tests (`tests/test_chunker.py`)

## Scripts

- `docsage/scripts/evaluate.py`: offline evaluation helper
- `docsage/scripts/fine_tune.py`: model/domain adaptation scaffold

## Current Limitations

- Session and document registries are primarily in-memory (not fully externalized).
- Multi-worker backend (`uvicorn --workers 2` in Docker backend image) can lead to state divergence unless shared state is moved to Redis/DB.
- Docker frontend build expects `nginx.conf`; ensure it exists and is aligned with deployment path.
- Repository currently includes large local artifacts (`venv`, `node_modules`, model cache) that should be excluded in production repositories.

## Suggested Next Steps

- Add persistent DB/Redis-backed state for sessions/doc metadata.
- Standardize Docker files + compose references.
- Expand integration and frontend test coverage.
- Add CI workflow for lint/test/build.
- Add auth and tenant-level access control for production use.

## Contributing

For local development workflow, testing standards, and PR expectations, see [`CONTRIBUTING.md`](./CONTRIBUTING.md).