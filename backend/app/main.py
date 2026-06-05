"""
main.py — Vietnamese Law RAG API v3.0 Entry Point.

Architecture:
    Endpoints:
        POST /v1/chat/completions  → OpenAI-compatible (streaming + sync)
        GET  /v1/models            → Model listing (for frontend auto-detect)
        POST /api/chat             → Legacy endpoint (same pipeline, simpler schema)
        GET  /health | /           → Health check
        GET  /metrics              → In-memory pipeline metrics snapshot

    Pipeline (per request):
        1. QueryReflector  — intent routing (7-intent taxonomy) + query rewriting
        2. Retriever       — intent-aware hybrid search (Qdrant + Local/Cohere rerank)
        3. Generator       — intent-aware streaming generation (MiMo v2.5 Pro via AsyncOpenAI)
        4. TavilyLegalSearcher — web fallback when local RAG scores are low

    Client strategy:
        Both QueryReflector and Generator share a single AsyncOpenAI client pointed
        at Xiaomi's OpenAI-compatible MiMo endpoint.  This avoids two separate
        HTTP connection pools and keeps key management in one place.

        MiMo OpenAI-compat endpoint:
            https://api.xiaomimimo.com/v1

    SSE protocol:
        Streaming responses follow OpenAI chunk format.  The pipeline's internal
        __META__ event is stripped from the SSE stream and attached to the last
        chunk as `x_meta` so the frontend can update source chips without parsing
        the text stream.

v3.0 changes vs v2:
    - Removed legacy_router / openai_router (now defined inline here).
    - Removed retriever_engine singleton (Retriever is instantiated in lifespan).
    - AsyncOpenAI replaces the legacy sync client; all service calls are now
      fully non-blocking.
    - CHITCHAT short-circuit in routes.handle_chat means greetings never hit
      Qdrant or Cohere.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.api.routes import handle_chat, handle_retrieve, get_metrics
from app.core.config import settings
from app.schemas import ChatRequest, ChatResponse, DocumentSource, RetrieveRequest
from app.services.conversation_memory import get_memory
from app.services.generator import Generator
from app.services.query_reflector import QueryReflector
from app.services.retriever import Retriever
from app.services.graph_retriever import GraphRAGRetriever
from app.services.web_searcher import web_searcher_engine
from app.services.local_reranker import LocalReranker
from app.services.parent_store import ParentStore

# ──────────────────────────────────────────────
# Structured Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("law-rag")


# ──────────────────────────────────────────────
# Module-level service singletons
# (populated during lifespan startup)
# ──────────────────────────────────────────────
_reflector: Optional[QueryReflector] = None
_retriever: Optional[Retriever] = None
_generator: Optional[Generator] = None
_graph_retriever: Optional[GraphRAGRetriever] = None
_local_reranker: Optional[LocalReranker] = None


# ──────────────────────────────────────────────
# FastAPI Lifespan (startup / shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialises all heavy resources once at startup and tears them down
    cleanly on shutdown.

    Resource lifecycle:
        AsyncOpenAI    — shares a single connection pool across reflector + generator.
        QueryReflector — wraps the LLM client for intent routing.
        Generator      — wraps the LLM client for streaming generation.
        SentenceTransformer (DEk21) — local embedding model for query encoding (~400MB RAM).
        AsyncQdrantClient — connection to Qdrant vector DB.
        LocalReranker  — local Vietnamese reranker (primary).
        cohere.AsyncClientV2 — Cohere reranker API client (fallback).
        Retriever      — wires all three above for hybrid search + rerank.
        web_searcher_engine — module-level singleton; no extra init needed here.
    """
    global _reflector, _retriever, _generator, _graph_retriever, _local_reranker

    logger.info("=" * 60)
    logger.info("Starting Vietnamese Law RAG API v3.1 ...")

    # ── Shared async LLM client (MiMo v2.5 Pro) ─────────────────────
    llm_client = AsyncOpenAI(
        api_key=settings.MIMO_API_KEY,
        base_url=settings.MIMO_BASE_URL,
    )

    # ── Pipeline services ────────────────────────────────────────
    _reflector = QueryReflector(
        llm_client=llm_client,
        model=settings.QUERY_REFLECT_MODEL,
    )
    _generator = Generator(
        client=llm_client,
        model=settings.LLM_MODEL,
        max_tokens=2048,
    )

    # ── Embedding model (DEk21 SentenceTransformer) ──────────────
    # Loaded once at startup. DEk21 is a lightweight 768-dim dense-only model.
    # Much smaller RAM footprint than BGE-M3 (~2GB → ~400MB).
    embedding_model = None
    try:
        # --- FIX: Auto-delete HuggingFace .lock files to prevent download hangs ---
        import os
        import glob
        hf_cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
        lock_files = glob.glob(f"{hf_cache_dir}/**/*.lock", recursive=True)
        if lock_files:
            logger.warning("Found %d HuggingFace .lock files. Deleting to prevent hang...", len(lock_files))
            for lf in lock_files:
                try:
                    os.remove(lf)
                except Exception:
                    pass
        # --------------------------------------------------------------------------

        from sentence_transformers import SentenceTransformer
        logger.info("Loading DEk21 embedding model: %s ...", settings.EMBEDDING_MODEL)
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("✅ DEk21 loaded successfully (dim=%d).", embedding_model.get_sentence_embedding_dimension())
    except Exception as exc:
        logger.error("❌ Failed to load DEk21: %s — retriever will degrade to web fallback", exc)

    # ── Qdrant async client ──────────────────────────────────────
    qdrant_client = None
    try:
        from qdrant_client import AsyncQdrantClient
        import asyncio
        
        qdrant_client = AsyncQdrantClient(url=settings.QDRANT_HOST, timeout=30.0)
        
        # Verify connectivity by listing collections with retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                collections = await qdrant_client.get_collections()
                col_names = [c.name for c in collections.collections]
                logger.info("✅ Qdrant connected: %s — collections: %s", settings.QDRANT_HOST, col_names)
                if settings.COLLECTION_NAME not in col_names:
                    logger.warning(
                        "⚠️  Collection '%s' not found in Qdrant. "
                        "Run migrate.py first to populate the vector DB.",
                        settings.COLLECTION_NAME,
                    )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning("⏳ Qdrant not ready, retrying in 3s... (%d/%d)", attempt + 1, max_retries)
                await asyncio.sleep(3)
                
    except Exception as exc:
        logger.error("❌ Qdrant connection failed: %s — retriever will degrade to web fallback", exc)

    # ── Cohere reranker client ───────────────────────────────────
    cohere_client = None
    if settings.COHERE_API_KEY:
        try:
            import cohere
            cohere_client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
            logger.info("✅ Cohere reranker client initialized.")
        except Exception as exc:
            logger.error("❌ Cohere client init failed: %s — reranking disabled", exc)
    else:
        logger.warning("⚠️  COHERE_API_KEY not set — reranking disabled (vector scores only).")

    # ── Local reranker ───────────────────────────────────────────
    if settings.USE_LOCAL_RERANKER:
        try:
            _local_reranker = LocalReranker(
                model_name=settings.RERANKER_MODEL,
                max_length=settings.RERANKER_MAX_LENGTH,
                request_max_length=settings.RERANKER_REQUEST_MAX_LENGTH,
            )
            _local_reranker.initialize()
        except Exception as exc:
            logger.error("❌ Local reranker init failed: %s", exc)
    else:
        logger.info("⚠️  Local reranker disabled (USE_LOCAL_RERANKER=false).")

    # ── Parent Store (SQLite lookup for parent_text) ────────────
    parent_store = None
    if settings.PARENTS_SQLITE_PATH:
        try:
            parent_store = ParentStore(settings.PARENTS_SQLITE_PATH)
            logger.info("✅ ParentStore loaded: %s", settings.PARENTS_SQLITE_PATH)
        except Exception as exc:
            logger.error("❌ ParentStore init failed: %s — retriever will lack parent_text", exc)

    # ── Retriever (fully wired) ──────────────────────────────────
    _retriever = Retriever(
        qdrant_client=qdrant_client,
        cohere_client=cohere_client,
        local_reranker=_local_reranker,
        embedding_model=embedding_model,
        parent_store=parent_store,
        collection_name=settings.COLLECTION_NAME,
    )

    # ── GraphRAG retriever (LanceDB) ────────────────────────────
    if settings.GRAPHRAG_ENABLED:
        _graph_retriever = GraphRAGRetriever(
            lancedb_path=settings.GRAPHRAG_LANCEDB_PATH,
            embedding_model=embedding_model,
            relationships_parquet=settings.GRAPHRAG_RELATIONSHIPS_PARQUET,
            local_reranker=_local_reranker,
        )
        await _graph_retriever.initialize()
    else:
        logger.info("⚠️  GraphRAG disabled (GRAPHRAG_ENABLED=false).")

    logger.info("✅ Pipeline services ready.")
    logger.info("   reflector model : %s", settings.QUERY_REFLECT_MODEL)
    logger.info("   generator model : %s", settings.LLM_MODEL)
    logger.info("   collection      : %s", settings.COLLECTION_NAME)
    logger.info("   embedder        : %s", "DEk21" if embedding_model else "NONE")
    logger.info("   local_reranker  : %s", "Vietnamese_Reranker" if _local_reranker else "NONE")
    logger.info("   cohere_reranker : %s", "Cohere" if cohere_client else "NONE")
    logger.info("   graphrag        : %s", "Enabled" if _graph_retriever else "Disabled")
    logger.info("=" * 60)

    yield   # ← Application is live

    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("Shutting down — closing clients ...")
    await llm_client.close()
    if qdrant_client:
        await qdrant_client.close()
    if cohere_client:
        try:
            await cohere_client.close()
        except AttributeError:
            pass
    logger.info("Shutdown complete.")


# ──────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────
app = FastAPI(
    title="Vietnamese Law RAG API",
    version="3.0.0",
    description=(
        "Hệ thống AI Tư vấn Luật Pháp Việt Nam — v3.0 Intent-Aware Pipeline.\n\n"
        "Tương thích chuẩn OpenAI API — Hỗ trợ tích hợp với ChatGPT Next Web, "
        "Lobe Chat, Open WebUI và các frontend khác."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ──────────────────────────────────────────────
# CORS Middleware
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request Timing Middleware
# ──────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request with method, path, and wall-clock time."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %s (%.0fms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ──────────────────────────────────────────────
# Helpers — dependency accessor
# ──────────────────────────────────────────────
def _get_services():
    """
    Return the singleton pipeline services, raising 503 if startup hasn't
    completed (e.g. health-check race on Kubernetes).
    """
    if _reflector is None or _generator is None or _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready. Startup may still be in progress.",
        )
    return _reflector, _retriever, _generator, _graph_retriever, _local_reranker


# ──────────────────────────────────────────────
# SSE streaming helpers
# ──────────────────────────────────────────────

async def _openai_sse_stream(
    gen: AsyncGenerator[str, None],
    model_id: str = "law-chatbot-v3",
) -> AsyncGenerator[str, None]:
    """
    Convert raw handle_chat tokens into OpenAI-format SSE events.

    The pipeline yields text tokens plus a single terminal __META__: event.
    This wrapper:
        - Buffers the __META__ payload.
        - Wraps each text token as a "chat.completion.chunk" SSE event.
        - Attaches the meta payload to the final [DONE] chunk as `x_meta`.

    Frontend consumers that don't understand x_meta simply ignore it.
    """
    meta_payload: Optional[Dict] = None

    async for token in gen:
        if token.startswith("__META__:"):
            try:
                meta_payload = json.loads(token[len("__META__:"):])
            except json.JSONDecodeError:
                meta_payload = {}
            continue  # don't emit meta as a text chunk

        chunk = {
            "object": "chat.completion.chunk",
            "model":  model_id,
            "choices": [{
                "index":         0,
                "delta":         {"content": token},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Final chunk — signals end of stream and carries the meta payload.
    final_chunk = {
        "object": "chat.completion.chunk",
        "model":  model_id,
        "choices": [{
            "index":         0,
            "delta":         {},
            "finish_reason": "stop",
        }],
        "x_meta": meta_payload or {},
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def _collect_full_response(
    gen: AsyncGenerator[str, None],
) -> tuple[str, Dict]:
    """
    Drain handle_chat, accumulate all text tokens, and extract the meta dict.

    Returns (answer_text, meta_dict) for non-streaming endpoints.
    """
    parts: List[str] = []
    meta: Dict = {}

    async for token in gen:
        if token.startswith("__META__:"):
            try:
                meta = json.loads(token[len("__META__:"):])
            except json.JSONDecodeError:
                pass
        else:
            parts.append(token)

    return "".join(parts), meta


# ──────────────────────────────────────────────
# OpenAI-compatible request schema
# ──────────────────────────────────────────────
class _OAIMessage(BaseModel):
    role:    str
    content: str

class _OAIRequest(BaseModel):
    model:    str = "law-chatbot-v3"
    messages: List[_OAIMessage]
    stream:   bool = True
    # Accept (and ignore) common OpenAI params so frontend SDKs don't error.
    temperature: Optional[float] = None
    max_tokens:  Optional[int]   = None
    # Optional conversation ID for memory tracking.
    # If not provided, a stable hash of the message list is used.
    conversation_id: Optional[str] = None


# ──────────────────────────────────────────────
# POST /v1/chat/completions
# ──────────────────────────────────────────────
@app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def openai_chat_completions(body: _OAIRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Extracts the last user message as the query.  Supports both streaming
    (SSE, text/event-stream) and non-streaming (JSON) modes.

    v3.1: Conversation memory is tracked per conversation_id.  The last
    WINDOW_SIZE turns are injected into both the reflector and generator.
    After generation, the completed turn is stored in memory.

    Streaming response format:
        data: {"object": "chat.completion.chunk", "choices": [...], ...}
        ...
        data: {"object": "chat.completion.chunk", "choices": [{"finish_reason": "stop"}],
               "x_meta": {"intent": "...", "sources": [...]}}
        data: [DONE]
    """
    import hashlib

    reflector, retriever, generator, graph_retriever, local_reranker = _get_services()

    # Extract the last user turn as the query.
    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=422, detail="No user message found in request.")
    user_query = user_messages[-1].content.strip()
    if not user_query:
        raise HTTPException(status_code=422, detail="User message is empty.")

    # ── Conversation memory ──
    # Derive a stable conversation ID: prefer explicit field, else hash of
    # all prior messages so the same chat thread maps to the same memory slot.
    if body.conversation_id:
        cid = body.conversation_id
    else:
        prior = "".join(m.role + m.content for m in body.messages[:-1])
        cid = hashlib.md5(prior.encode(), usedforsecurity=False).hexdigest() if prior else "default"

    memory = await get_memory(
        conversation_id=cid,
        llm_client=generator._client,
        model=settings.QUERY_REFLECT_MODEL,
    )
    history_messages = await memory.get_context_messages()

    logger.debug(
        "[main] cid=%s history_turns=%d",
        cid, len([m for m in history_messages if m["role"] == "user"]),
    )

    pipeline_gen = handle_chat(
        user_query=user_query,
        reflector=reflector,
        retriever=retriever,
        generator=generator,
        web_searcher=web_searcher_engine,
        stream=body.stream,
        history_messages=history_messages,
        graph_retriever=graph_retriever,
        local_reranker=local_reranker,
    )

    if body.stream:
        # For streaming: wrap generator so we can capture the full answer
        # and store it in memory after the stream completes.
        async def _streaming_with_memory() -> AsyncGenerator[str, None]:
            collected_tokens: List[str] = []
            async for event in _openai_sse_stream(pipeline_gen, model_id=body.model):
                yield event
                # Extract text from chunk for memory recording
                if event.startswith("data: ") and not event.startswith("data: [DONE]"):
                    try:
                        chunk_data = json.loads(event[6:])
                        delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected_tokens.append(content)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
            # Store completed turn after stream finishes
            answer_text = "".join(collected_tokens)
            if answer_text:
                await memory.add_turn(user_query, answer_text)
                logger.debug("[main] Stored turn in memory cid=%s", cid)

        return StreamingResponse(
            _streaming_with_memory(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: collect, store turn, then return JSON.
    answer, meta = await _collect_full_response(pipeline_gen)
    if answer:
        await memory.add_turn(user_query, answer)
        logger.debug("[main] Stored turn in memory cid=%s", cid)

    sources = [
        DocumentSource(
            id=s.get("id", ""),
            title=s.get("title", ""),
            url=s.get("url", ""),
            score=s.get("score", 0.0),
            entity=s.get("entity", ""),
        )
        for s in meta.get("sources", [])
    ]
    return {
        "object":  "chat.completion",
        "model":   body.model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "x_meta": {
            "intent":  meta.get("intent", "LEGAL_LOOKUP"),
            "sources": [s.model_dump() for s in sources],
        },
    }


# ──────────────────────────────────────────────
# GET /v1/models
# ──────────────────────────────────────────────
@app.get("/v1/models", tags=["OpenAI Compatible"])
async def list_models():
    """
    Return available models in OpenAI-compatible format.
    Frontend UIs (ChatGPT Next Web, Lobe Chat) query this to auto-detect models.
    """
    return {
        "object": "list",
        "data": [{
            "id":         "law-chatbot-v3",
            "object":     "model",
            "created":    1700000000,
            "owned_by":   "law-rag-system",
            "permission": [],
            "root":       "law-chatbot-v3",
            "parent":     None,
        }],
    }


# ──────────────────────────────────────────────
# POST /api/chat  (Legacy)
# ──────────────────────────────────────────────
@app.post("/api/chat", tags=["Legacy"])
async def legacy_chat(body: ChatRequest):
    """
    Legacy endpoint — same v3 pipeline, simpler request/response schema.

    v3.1: Tracks conversation memory via body.conversation_id.
    Always non-streaming; returns ChatResponse JSON.
    Use /v1/chat/completions for streaming.
    """
    reflector, retriever, generator, graph_retriever, local_reranker = _get_services()

    # ── Conversation memory ──
    cid = body.conversation_id or "legacy-default"
    memory = await get_memory(
        conversation_id=cid,
        llm_client=generator._client,
        model=settings.QUERY_REFLECT_MODEL,
    )
    history_messages = await memory.get_context_messages()

    pipeline_gen = handle_chat(
        user_query=body.message,
        reflector=reflector,
        retriever=retriever,
        generator=generator,
        web_searcher=web_searcher_engine,
        stream=False,
        include_content=body.include_content,
        top_k=body.top_k,
        history_messages=history_messages,
        graph_retriever=graph_retriever,
        local_reranker=local_reranker,
    )

    answer, meta = await _collect_full_response(pipeline_gen)

    if answer:
        await memory.add_turn(body.message, answer)

    sources = [
        DocumentSource(
            id=s.get("id", ""),
            title=s.get("title", ""),
            url=s.get("url", ""),
            score=s.get("score", 0.0),
            entity=s.get("entity", ""),
            content=s.get("content", None),
        )
        for s in meta.get("sources", [])
    ]
    from_web = any(s.url for s in sources)

    return ChatResponse(
        answer=answer,
        sources=sources,
        intent=meta.get("intent", "LEGAL_LOOKUP"),
        confidence=0.0,     # not surfaced to legacy consumers
        from_web=from_web,
    )


# ──────────────────────────────────────────────
# POST /api/retrieve  (Retrieval-only)
# ──────────────────────────────────────────────
@app.post("/api/retrieve", tags=["Legacy"])
async def retrieve_only(body: RetrieveRequest):
    """
    Retrieval-only endpoint — runs intent routing + hybrid search + rerank,
    but skips the LLM generation step entirely.

    Intended for:
      - Evaluation scripts that generate answers independently.
      - Debugging retrieval quality without paying for generation.
      - Any client that only needs ranked source documents.

    Response JSON:
        {
          "intent":        str,           # detected intent
          "refined_query": str,           # rewritten query sent to Qdrant
          "confidence":    float,         # intent classification confidence
          "sources":       List[Source],  # ranked docs (content included by default)
          "retrieval_ms":  float,         # wall-clock retrieval time in ms
          "used_web":      bool           # true if Tavily fallback was triggered
        }
    """
    reflector, retriever, _, graph_retriever, local_reranker = _get_services()

    result = await handle_retrieve(
        user_query=body.message,
        reflector=reflector,
        retriever=retriever,
        web_searcher=web_searcher_engine,
        top_k=body.top_k,
        include_content=body.include_content,
        graph_retriever=graph_retriever,
        local_reranker=local_reranker,
    )

    # Normalise sources to DocumentSource objects for schema validation.
    result["sources"] = [
        DocumentSource(
            id=s.get("id", ""),
            title=s.get("title", ""),
            url=s.get("url", ""),
            score=s.get("score", 0.0),
            entity=s.get("entity", ""),
            content=s.get("content") if body.include_content else None,
        ).model_dump(exclude_none=True)
        for s in result["sources"]
    ]
    return result


# ──────────────────────────────────────────────
# GET /health | /
# ──────────────────────────────────────────────
@app.get("/", tags=["System"])
@app.get("/health", tags=["System"])
async def health_check():
    """Returns system status and available endpoints."""
    ready = _reflector is not None and _generator is not None
    return {
        "status":  "ok" if ready else "starting",
        "version": "3.0.0",
        "message": "Vietnamese Law RAG API v3.0 đang hoạt động!",
        "pipeline": {
            "intent_taxonomy": [
                "LEGAL_LOOKUP", "COMPARE", "SUMMARIZE",
                "PROCEDURE", "DEFINITION", "MULTI_HOP", "CHITCHAT",
            ],
            "web_fallback": "tavily",
        },
        "endpoints": {
            "openai_chat":  "POST /v1/chat/completions",
            "models":       "GET  /v1/models",
            "legacy_chat":  "POST /api/chat",
            "metrics":      "GET  /metrics",
            "docs":         "GET  /docs",
        },
    }


# ──────────────────────────────────────────────
# GET /metrics
# ──────────────────────────────────────────────
@app.get("/metrics", tags=["System"])
async def pipeline_metrics():
    """
    Return a snapshot of the in-memory pipeline metrics.

    Includes:
      - total_requests, legal_requests, chitchat_requests
      - web_fallbacks, errors
      - per-intent counters (LEGAL_LOOKUP, COMPARE, SUMMARIZE, ...)

    Note: these reset on process restart.
    Migrate to Prometheus/StatsD for persistent metrics in production.
    """
    return get_metrics()
