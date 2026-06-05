# Vietnamese Law Chatbot

A production-oriented Vietnamese legal assistant built around retrieval-augmented generation (RAG). The system answers legal questions by routing each query through intent classification, legal document retrieval, reranking, optional GraphRAG expansion, trusted web fallback, and LLM generation with source metadata.

## What It Can Do

- Answer Vietnamese legal questions with citations from retrieved legal sources.
- Classify user intent into legal lookup, comparison, summarization, procedure, definition, multi-hop, and chitchat flows.
- Retrieve legal chunks from Qdrant using DEk21 dense embeddings and intent-aware search strategies.
- Rerank retrieved evidence with a local Vietnamese reranker, with Cohere available as a fallback.
- Expand complex questions with GraphRAG data stored in LanceDB and relationship parquet files.
- Fall back to trusted Vietnamese legal websites through Tavily when local retrieval confidence is low.
- Serve OpenAI-compatible chat endpoints for frontends such as ChatGPT Next Web, Lobe Chat, or Open WebUI.
- Preserve short conversation memory so follow-up legal questions can use recent context.
- Evaluate retrieval and answer quality with a separate evaluation worker and RAGAS-style tooling.

## What We Built

This repository contains an end-to-end legal AI application:

- `backend/`: FastAPI service exposing `/v1/chat/completions`, `/v1/models`, `/api/chat`, `/api/retrieve`, `/health`, and `/metrics`.
- `backend/app/services/`: core RAG services for query reflection, retrieval, reranking, GraphRAG retrieval, generation, conversation memory, and trusted web search.
- `data_pipelines/vector_rag/`: vector ingestion and metadata enrichment pipelines for Qdrant.
- `data_pipelines/graph_rag/`: GraphRAG embedding and LanceDB preparation scripts.
- `evaluation/`: golden dataset creation and evaluation scripts for retrieval and answer quality checks.
- `chatbot-ui/NextChat/`: frontend client integration based on ChatGPT Next Web.
- `docker-compose.yml`: local orchestration for Qdrant, backend API, frontend UI, evaluation worker, and GraphRAG embedding worker.

## Technical Skills Demonstrated

- Python backend engineering with FastAPI, Pydantic, async service orchestration, streaming responses, and health/metrics endpoints.
- LLM application design with OpenAI-compatible APIs, streaming SSE, prompt-aware generation, query rewriting, and conversation memory.
- RAG system design with Qdrant vector search, dense embeddings, reranking, parent-child chunk expansion, source formatting, and confidence-based fallback.
- Vietnamese NLP integration with DEk21 embeddings, Vietnamese tokenization, and a local Vietnamese reranker.
- GraphRAG data engineering with entity/community embeddings, LanceDB, relationship parquet loading, and hybrid graph/vector retrieval.
- Data pipeline engineering for legal corpus ingestion, SQLite parent stores, metadata enrichment, checkpointing, and Dockerized workers.
- Evaluation engineering with golden dataset construction, retrieval-only APIs, RAGAS-style evaluation, and repeatable evaluation containers.
- DevOps and deployment with Docker Compose, service profiles, resource limits, mounted model caches, and environment-driven configuration.
- Frontend/API integration using an OpenAI-compatible contract so existing chat UIs can talk to the custom law backend.

## Architecture

```text
User / Chat UI
    |
    v
FastAPI backend
    |
    +-- QueryReflector: intent detection, query rewriting, expanded queries
    |
    +-- Route selection:
    |      LEGAL_LOOKUP, COMPARE, SUMMARIZE, PROCEDURE, DEFINITION, MULTI_HOP
    |
    +-- Retrieval:
    |      Qdrant + DEk21 dense embeddings
    |      GraphRAG via LanceDB when enabled
    |      ParentStore SQLite enrichment
    |
    +-- Reranking:
    |      Local Vietnamese_Reranker first
    |      Cohere rerank fallback when configured
    |
    +-- Fallback:
    |      Tavily search over trusted Vietnamese legal domains
    |
    v
Generator: Vietnamese legal answer with source metadata
```

## Main API Surface

- `POST /v1/chat/completions`: OpenAI-compatible chat endpoint with streaming and non-streaming modes.
- `GET /v1/models`: model listing for compatible frontend clients.
- `POST /api/chat`: simpler legacy chat endpoint.
- `POST /api/retrieve`: retrieval-only endpoint for debugging and evaluation.
- `GET /health`: service readiness and endpoint summary.
- `GET /metrics`: in-memory pipeline counters by request type and intent.

## Local Run

Create a `.env` file with the API keys and model settings required by your environment, then start the stack:

```bash
docker compose up -d
```

The main services are:

- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Frontend UI: `http://localhost:3000`
- Qdrant: `http://localhost:6333`

Optional workers are available through Compose profiles:

```bash
docker compose --profile eval run --rm evaluation-worker python build_golden_dataset.py --help
docker compose --profile graphrag run --rm graphrag-embedder
```

## Configuration

Important environment variables include:

- `MIMO_API_KEY`, `MIMO_BASE_URL`, `LLM_MODEL`, `QUERY_REFLECT_MODEL`
- `QDRANT_HOST`, `COLLECTION_NAME`
- `USE_LOCAL_RERANKER`, `RERANKER_MODEL`, `RERANKER_REQUEST_MAX_LENGTH`
- `COHERE_API_KEY`
- `TAVILY_API_KEY`, `RAG_FALLBACK_SCORE_THRESHOLD`
- `GRAPHRAG_ENABLED`, `GRAPHRAG_LANCEDB_PATH`, `GRAPHRAG_RELATIONSHIPS_PARQUET`
- `PARENTS_SQLITE_PATH`

## Notes

Large local datasets, model caches, generated GraphRAG outputs, `.env`, virtual environments, and agent-tooling folders are intentionally ignored. The repository should contain source code, pipeline definitions, documentation, and reproducible scripts rather than machine-local state.
