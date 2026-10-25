# KNUST Admission GPT

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions about **Kwame Nkrumah University of Science and Technology (KNUST)** undergraduate admissions — entry requirements, programmes, cut-off aggregates, application steps, and fees.

The assistant is grounded in KNUST's official admissions document and cites its sources. If an answer isn't in the source material, it says so rather than guessing.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│    FastAPI       │────▶│  Pinecone       │
│   (UI)      │     │   (Backend)      │     │  (Vector DB)    │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                    ┌──────────────────┐              │
                    │   OpenAI         │◀─────────────┘
                    │   (Embeddings +  │
                    │    Generation)   │
                    └──────────────────┘
```

### Retrieval Pipeline (AdvancedRetriever)

The retriever combines multiple techniques for high precision:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | **Query Rewrite** | Normalize messy/conversational questions into clean search queries |
| 2 | **Multi-Query Expansion** | Generate alternative phrasings to improve recall |
| 3 | **Hybrid Search** | Dense (Pinecone) + Sparse (BM25) retrieval per phrasing |
| 4 | **Reciprocal Rank Fusion (RRF)** | Fuse all result lists into one ranked pool |
| 5 | **Cross-Encoder Rerank** | Pinecone-hosted `bge-reranker-v2-m3` for final precision |

### Key Features

- **Conversational context** — Follow-up questions ("what about the Obuasi campus version?") are resolved into standalone queries before retrieval
- **Source citations** — Every answer includes the retrieved chunks with relevance scores
- **Quality testing** — RAGAS-based regression suite (`context_recall`, `faithfulness`, `answer_relevancy`)
- **Cost transparency** — Token counting and estimated embedding costs logged during indexing

---

## Project Structure

```
rag_project/
├── rag_deploy/                 # Production deployment code
│   ├── main.py                 # FastAPI backend (POST /query, GET /health)
│   ├── retrieval_pipeline.py   # AdvancedRetriever + generation logic
│   ├── indexing_pipeline.py    # PDF → chunks → embeddings → Pinecone
│   └── streamlit_app.py        # Chat UI (talks to FastAPI via HTTP)
├── rag_test/                   # Quality regression suite
│   ├── test_rag_quality.py     # RAGAS metrics on golden set
│   ├── retrieval_pipeline.py   # Test-specific retriever config
│   ├── golden_set.json         # Q/A pairs with ground truth
│   └── conftest.py             # Pytest fixtures
├── rag_app/                    # Notebook exploration
│   ├── rag_indexing_pipeline.ipynb
│   └── rag_retrieval_pipeline.ipynb
├── simple_rag_app/             # Simpler prototypes (OpenAI / Ollama)
├── data/
│   └── admission_requirement.pdf  # Source document
├── overview/                   # Project documentation / course materials
├── .env                        # Environment variables (not committed)
├── pyproject.toml              # Dependencies (uv)
├── requirements.txt            # Minimal pip dependencies
└── uv.lock                     # Locked dependency versions
```

---

## Prerequisites

- **Python 3.13+** (the deployment pin is 3.13; local checks used 3.13.16)
- **Pinecone account** (serverless indexes)
- **OpenAI API key** (embeddings + chat)
- **uv** (recommended) or pip

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd rag_project

# Using uv (recommended)
uv sync

# Or with pip
python -m pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` in the project root and fill in your keys:

```env
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
INDEX_NAME=knust-admission-rag
SPARSE_INDEX_NAME=knust-rag-sparse
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
CHAT_MODEL=gpt-4o-mini
```

### 3. Build the index (run once)

```bash
# From the repository root; install the PDF loader first
uv sync --extra indexing
uv run python -m rag_deploy.indexing_pipeline
```

This will:
- Load and clean the KNUST admissions PDF
- Chunk with `RecursiveCharacterTextSplitter` (500 chars, 50 overlap)
- Add page context passages (up to 3,000 chars, 500 overlap) to keep shared programme requirements together
- Preserve numeric values and add one dated passage per programme in the cutoff tables
- Embed with `text-embedding-3-small`
- Create two Pinecone indexes: dense (cosine) + sparse (BM25)
- Upsert all chunks to both indexes

Skip this step if both indexes already contain your admissions data. Existing
indexes are reused; indexing does not delete them.
Set `PINECONE_NAMESPACE` to a new name for an isolated test dataset, and use the
same value for indexing and the backend. Reusing a namespace upserts matching
IDs but does not remove stale chunks from earlier builds.

### 4. Start the backend

```bash
# From the repository root
uv run python -m uvicorn rag_deploy.main:app --reload --port 8000
```

API docs available at `http://127.0.0.1:8000/docs`

### 5. Launch the UI

```bash
# From the repository root (in a new terminal)
uv run python -m streamlit run rag_deploy/streamlit_app.py
```

Open `http://localhost:8501` and start asking questions.

For pip installations, activate your virtual environment and omit `uv run`.
`GET /health` checks that the API process is running; a successful `/query`
also requires valid credentials and populated Pinecone indexes. Restart the
backend after changing `.env`.

On Windows, you can use the installed environment directly in two terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn rag_deploy.main:app --port 8000
.\.venv\Scripts\python.exe -m streamlit run rag_deploy/streamlit_app.py
```

The backend uses the Windows certificate trust store for verified HTTPS
connections. Use Python 3.13.16 or a newer 3.13 patch locally to avoid the older bundled runtime's
[`OPENSSL_Applink` crash](https://github.com/astral-sh/python-build-standalone/issues/640).

---

## Share the pilot: Vercel API + Streamlit Community Cloud UI

The Streamlit page calls a separate FastAPI service. Deploy `rag_deploy.main:app`
to Vercel first, then point the Streamlit Community Cloud app at its public URL.
The Vercel project uses the root `pyproject.toml` and `uv.lock`; Streamlit Cloud
uses the smaller `rag_deploy/requirements.txt` beside its entrypoint. Vercel's
Python runtime supports FastAPI and Python 3.13, while Streamlit Cloud lets you
select Python 3.13 when creating the app. See the [Vercel FastAPI guide](https://vercel.com/docs/frameworks/backend/fastapi)
and [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

1. Review and push the source changes to this repository's GitHub `main` branch.
   `.env`, `.env.local`, and `.streamlit/secrets.toml` are ignored. Do not add
   them to Git. The local validation findings in
   [`rag_test/LOCAL_VALIDATION.md`](rag_test/LOCAL_VALIDATION.md) describe known
   answer errors; share the app as a pilot for feedback.
2. In [Vercel](https://vercel.com/new), import `mrdeecy/KNUSTAdmissionGPT`
   with the repository root as the project directory and FastAPI as the
   framework. `pyproject.toml` identifies `rag_deploy.main:app`; `vercel.json`
   allows 120 seconds for a query and excludes non-runtime files. Do not set a
   custom build or output directory.
3. Under Vercel **Project Settings → Environment Variables**, set
   `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_NAMESPACE`, and
   `KNUST_APP_TOKEN` for Production. Use
   `PINECONE_NAMESPACE=local-validation-20261024-v3` to query the exact corpus
   validated locally. Use a long random token for `KNUST_APP_TOKEN` and keep
   it private. Set `INDEX_NAME` and `SPARSE_INDEX_NAME` too if your indexes have
   non-default names. Redeploy after adding or changing variables. Vercel's
   [environment variable guide](https://vercel.com/docs/environment-variables)
   explains environment scoping.
4. Confirm `https://YOUR-VERCEL-DOMAIN/health` returns `{"status":"ok"}`.
   `/health` only checks that the process starts; test `/query` with the
   `X-App-Token` header to verify the OpenAI and Pinecone connections.
5. At [share.streamlit.io](https://share.streamlit.io), create an app from the
   same GitHub repo and branch. Set **Main file path** to
   `rag_deploy/streamlit_app.py` and choose **Python 3.13** under Advanced
   settings. In the Secrets field, enter only these two root-level TOML values:

   ```toml
   KNUST_API_URL = "https://YOUR-VERCEL-DOMAIN"
   KNUST_APP_TOKEN = "THE-SAME-PRIVATE-TOKEN-AS-VERCEL"
   ```

   Streamlit exposes root-level secrets as environment variables to the app.
   The OpenAI and Pinecone keys belong only in Vercel. See [Streamlit secrets
   management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
6. Open the Streamlit URL and ask the Computer Science requirements question,
   followed by “What is its cut-off aggregate?” Check that both answers cite
   sources. Share that Streamlit URL with testers.

The Vercel API requires `X-App-Token` when `KNUST_APP_TOKEN` is configured.
The Streamlit server sends the token on each query; it does not appear in the
browser. Keep the token and provider keys out of the repository. In local
development, leaving `KNUST_APP_TOKEN` unset preserves the previous workflow.

---

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

### `POST /query`

Ask a question.

**Request:**
```json
{
  "question": "What are the WASSCE requirements for BSc Computer Science?",
  "top_k": 3,
  "history": [
    { "role": "user", "content": "Tell me about engineering programmes" },
    { "role": "assistant", "content": "KNUST offers several engineering programmes..." }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | The applicant's question |
| `top_k` | integer | No | Number of cited chunks (1-10, default: retriever's configured value) |
| `history` | array | No | Prior conversation turns for follow-up resolution |

**Response:**
```json
{
  "answer": "For BSc Computer Science, WASSCE applicants need... [1][2]",
  "sources": [
    {
      "chunk_id": "KNUST_admission_requirements_pdf_005",
      "title": "Faculty of Computing and Mathematical Sciences",
      "text": "BSc Computer Science: WASSCE passes in English, Core Mathematics...",
      "score": 0.847
    }
  ]
}
```

---

## Running Tests

### Unit / Integration Tests (fast, no LLM)

```bash
# From the repository root
uv run --with pytest python -m pytest -v
```

### RAG Quality Regression (requires LLM + Pinecone)

Start the FastAPI backend first. The evaluator calls its `/query` endpoint
(`KNUST_API_URL`, default `http://127.0.0.1:8000`) using the same retrieval
settings as the UI. RAGAS 0.3.1 needs older LangChain integrations, so install
its pinned dependencies in a separate environment instead of the app environment.

```powershell
uv venv .tools/ragas-env --python 3.13.16
uv pip install --python .tools/ragas-env/Scripts/python.exe -r rag_test/evaluation-requirements.txt
.\.tools\ragas-env\Scripts\python.exe -m pytest rag_test/test_rag_quality.py -m llm -v -o junit_family=legacy --junitxml=.local/ragas-results.xml
```

In this workspace, use `.\.tools\uv.exe` if `uv` is not on PATH. On macOS/Linux,
use `bin/python` in place of `Scripts/python.exe`. The XML report includes each
question, answer, retrieved sources, and metric scores. Non-finite scores fail.

Local tests exercise the production API, retrieval orchestration, dense-vector
upload, and Streamlit conversation flow using mocked external services.
Live quality tests exercise the running production API and assert RAGAS metrics:
- `context_recall >= 0.5`
- `faithfulness >= 0.5`
- `answer_relevancy >= 0.5`

---

## Configuration

Key environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEX_NAME` | `knust-admission-rag` | Pinecone dense index name |
| `SPARSE_INDEX_NAME` | `knust-rag-sparse` | Pinecone sparse (BM25) index name |
| `PINECONE_NAMESPACE` | `__default__` | Shared namespace for dense and sparse records |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIM` | `1536` | Embedding dimensions |
| `CHAT_MODEL` | `gpt-4o-mini` | Generation model |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | Pinecone hosted reranker |

Retriever knobs (in `AdvancedRetriever.__init__`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | `3` | Final chunks returned after rerank |
| `candidates` | `10` | Candidates per search before RRF |

---

## Data Source

The knowledge base is built from **KNUST's official undergraduate admissions requirements document** (`data/admission_requirement.pdf`), covering:

- Entry requirements by qualification (WASSCE, A-Level, IB, HND, Mature)
- Programme-specific subject combinations and cut-off aggregates
- Application procedures, deadlines, and fees
- Campus-specific requirements (Main campus, Obuasi, etc.)

---

## Deployment Notes

### Backend (FastAPI)
- Stateless — scale horizontally behind a load balancer
- Set `KNUST_API_URL` in the Streamlit config to point to the deployed backend
- CORS enabled for all origins (`allow_origins=["*"]`)

### Frontend (Streamlit)
- Pure client-side — talks to backend via HTTP only
- Deploy to Streamlit Community Cloud, Render, Railway, etc.

### Vector Database (Pinecone)
- Serverless indexes (AWS us-east-1)
- Dense index: cosine similarity, 1536 dimensions
- Sparse index: Pinecone integrated BM25 (`pinecone-sparse-english-v0`)

---

## Cost Estimation

| Operation | Model | Cost (approx) |
|-----------|-------|---------------|
| Embedding (indexing) | text-embedding-3-small | $0.02 / 1M tokens |
| Embedding (query) | text-embedding-3-small | ~500 tokens/query |
| Generation | gpt-4o-mini | $0.15 / 1M input, $0.60 / 1M output |
| Rerank | bge-reranker-v2-m3 | Included in Pinecone serverless |

Typical query: ~2K input tokens (context + history) + ~300 output tokens ≈ **$0.001/query**

---

## License

MIT License — see `LICENSE` file for details.

---

## Acknowledgments

- **Pinecone** for vector database and hosted reranking
- **OpenAI** for embeddings and generation models
- **LangChain** for document loading and text splitting
- **RAGAS** for quality evaluation framework
- **KNUST** for the admissions reference document
