"""
routes.py — API route handlers.

v3.0 changes:
  - `intent` and `sub_entities` from ReflectionResult are threaded through
    to both the retriever and the generator.
  - `ChatResponse` includes the `intent` field (see schemas.py).
  - Metrics track all 7 intent types; non-CHITCHAT are also counted as
    "legal" for backward-compatible aggregate counters.
  - CHITCHAT short-circuit: returns immediately without calling the retriever
    (saves vector DB + reranker API cost).
  - Mean-score fallback threshold check applies to the merged COMPARE pool
    before Tavily is invoked.

v3.1 changes:
  - handle_chat() and handle_retrieve() now accept optional history_messages
    (List[Dict]) from ConversationMemory and pass it to reflector.reflect()
    and generator.generate() for full multi-turn awareness.

Key logical note on fallback triggering:
  The existing fallback condition (`mean_score < threshold`) is evaluated
  on the reranked docs.  For COMPARE, this correctly reflects whether
  BOTH entities have good coverage.  For MULTI_HOP, a low mean score after
  chaining indicates the knowledge base may not cover the full chain —
  web fallback is appropriate.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncio

from app.core.config import settings
from app.services.query_reflector import IntentType, ReflectionResult, is_legal_intent
from app.services.query_router import route_retrieval, get_route

log = logging.getLogger("pipeline")

# Reranked-score threshold below which Tavily web search is triggered.
# Reads from settings so it can be tuned via RAG_FALLBACK_SCORE_THRESHOLD env var
# without redeploying code (default: 0.35 set in config.py).
_FALLBACK_SCORE_THRESHOLD: float = settings.RAG_FALLBACK_SCORE_THRESHOLD

# Metrics counters (in-memory; replace with Prometheus/StatsD in production).
_METRICS: Dict[str, int] = {
    "total_requests":   0,
    "legal_requests":   0,
    "chitchat_requests": 0,
    "web_fallbacks":    0,
    "errors":           0,
    "last_error":       "",
    # Use __members__ to iterate canonical names and avoid the
    # LEGAL_QUERY alias (which equals LEGAL_LOOKUP) from excluding
    # LEGAL_LOOKUP via identity check.
    **{name: 0 for name in IntentType.__members__},
}


def _increment(key: str, n: int = 1) -> None:
    _METRICS[key] = _METRICS.get(key, 0) + n


# ---------------------------------------------------------------------------
# Core pipeline orchestrator
# ---------------------------------------------------------------------------

async def handle_retrieve(
    user_query: str,
    reflector: Any,
    retriever: Any,
    web_searcher: Any,
    top_k: int = 5,
    include_content: bool = True,
    history_messages: Optional[List[Dict]] = None,
    graph_retriever: Any = None,
    local_reranker: Any = None,
) -> Dict:
    """
    Retrieval-only pipeline — no generation.

    Runs Phase 1 (intent routing) and Phase 3 (retrieval + rerank + optional
    web fallback), then returns a structured dict ready for the evaluator or
    any client that only needs document sources.

    Returns
    -------
    dict with keys:
        intent         : str
        refined_query  : str
        confidence     : float
        sources        : List[Dict]  — same schema as /api/chat sources
        retrieval_ms   : float       — wall-clock time for retrieval phase only
        used_web       : bool
    """
    t0 = time.perf_counter()
    _increment("total_requests")

    # Phase 1: intent routing (history-aware)
    t_reflect = time.perf_counter()
    reflection: ReflectionResult = await reflector.reflect(
        user_query, history_messages=history_messages
    )
    reflect_ms = (time.perf_counter() - t_reflect) * 1000
    intent     = reflection.intent
    intent_val = intent.value if hasattr(intent, "value") else str(intent)
    _increment(intent_val)

    if not is_legal_intent(intent):
        _increment("chitchat_requests")
        log.info(
            "[routes] handle_retrieve: CHITCHAT  reflect_ms=%.0f",
            reflect_ms,
        )
        return {
            "intent":        intent_val,
            "refined_query": user_query,
            "confidence":    reflection.confidence,
            "sources":       [],
            "retrieval_ms":  0.0,
            "used_web":      False,
        }

    _increment("legal_requests")

    # Phase 3: routed retrieval (RAG / GraphRAG / both)
    retrieval_query = reflection.refined_query or user_query
    route = get_route(intent)
    t_ret = time.perf_counter()
    try:
        docs = await route_retrieval(
            query=retrieval_query,
            intent=intent,
            rag_retriever=retriever,
            graph_retriever=graph_retriever,
            top_k=top_k,
            extra_queries=reflection.expanded_queries,
            intent_val=intent_val,
            sub_entities=reflection.sub_entities,
            local_reranker=local_reranker,
        )
    except Exception as exc:
        log.error("[routes] handle_retrieve: retrieval failed: %s", exc)
        log.error("[routes] handle_retrieve traceback:\n%s", traceback.format_exc())
        _increment("errors")
        _METRICS["last_error"] = f"handle_retrieve: {exc}"
        docs = []
    retrieval_ms = (time.perf_counter() - t_ret) * 1000

    # Phase 4: optional web fallback (same logic as handle_chat)
    used_web = False
    if _should_fallback(docs):
        log.info("[routes] handle_retrieve: low scores — web fallback")
        _increment("web_fallbacks")
        used_web = True
        try:
            web_docs = await _web_fallback(retrieval_query, web_searcher)
            docs = web_docs or docs
        except Exception as exc:
            log.warning("[routes] handle_retrieve: web fallback failed: %s", exc)

    total_ms = (time.perf_counter() - t0) * 1000
    sources = _format_sources(docs, include_content=include_content)
    log.info(
        "[routes] handle_retrieve: intent=%s route=%s docs=%d web=%s "
        "reflect_ms=%.0f retrieve_ms=%.0f total_ms=%.0f",
        intent_val, route, len(docs), used_web,
        reflect_ms, retrieval_ms, total_ms,
    )
    return {
        "intent":        intent_val,
        "refined_query": retrieval_query,
        "confidence":    reflection.confidence,
        "sources":       sources,
        "retrieval_ms":  round(retrieval_ms, 1),
        "used_web":      used_web,
    }


async def handle_chat(
    user_query: str,
    reflector: Any,
    retriever: Any,
    generator: Any,
    web_searcher: Any,
    stream: bool = True,
    include_content: bool = False,
    top_k: int = 5,
    history_messages: Optional[List[Dict]] = None,
    graph_retriever: Any = None,
    local_reranker: Any = None,
) -> AsyncGenerator[str, None]:
    """
    Orchestrate one full query through the pipeline.

    Phases:
      1. Intent routing + query rewriting (QueryReflector).
      2. CHITCHAT short-circuit: skip retrieval, stream direct response.
      3. Intent-aware retrieval (Retriever).
      4. Web fallback if reranked scores are too low.
      5. Intent-aware generation (Generator).

    Yields
    ------
    SSE-compatible text tokens (or the full response string if stream=False).

    Also yields a final structured event token prefixed with "__META__:" that
    the SSE layer can strip and use to update the response object:
        __META__:{"intent": "COMPARE", "sources": [...]}

    This keeps the streaming protocol clean — the meta event is always last.
    """
    t_start = time.perf_counter()
    _increment("total_requests")

    # ── Phase 1: Intent routing (history-aware) ──
    t_reflect = time.perf_counter()
    reflection: ReflectionResult = await reflector.reflect(
        user_query, history_messages=history_messages
    )
    reflect_ms = (time.perf_counter() - t_reflect) * 1000
    intent     = reflection.intent
    intent_val = intent.value if hasattr(intent, "value") else str(intent)

    _increment(intent_val)

    log.info(
        "[routes] Query: %.80r  →  intent=%s  confidence=%.2f  source=%s  reflect_ms=%.0f",
        user_query, intent_val, reflection.confidence, reflection.source, reflect_ms,
    )

    # ── Phase 2: CHITCHAT short-circuit ──
    if not is_legal_intent(intent):
        _increment("chitchat_requests")
        yield reflection.response or "Xin chào! Tôi có thể giúp gì cho bạn?"
        yield f"__META__:{{\"intent\": \"{intent_val}\", \"sources\": []}}"
        return

    _increment("legal_requests")

    # ── Phase 3: Routed retrieval (RAG / GraphRAG / both) ──
    retrieval_query = reflection.refined_query or user_query
    route = get_route(intent)
    log.info("[routes] Route: %s → %s", intent_val, route)
    t_retrieve = time.perf_counter()
    try:
        docs = await asyncio.wait_for(
            route_retrieval(
                query=retrieval_query,
                intent=intent,
                rag_retriever=retriever,
                graph_retriever=graph_retriever,
                top_k=top_k,
                extra_queries=reflection.expanded_queries,
                intent_val=intent_val,
                sub_entities=reflection.sub_entities,
                local_reranker=local_reranker,
            ),
            timeout=120.0,  # 120s — 5.75M points + CPU DEk21 + reranker can be slow
        )
    except asyncio.TimeoutError:
        log.error("[routes] Retrieval timed out after 120s")
        _increment("errors")
        _METRICS["last_error"] = "Retrieval timed out after 120s"
        docs = []
    except Exception as exc:
        log.error("[routes] Retrieval failed: %s", exc)
        log.error("[routes] Retrieval traceback:\n%s", traceback.format_exc())
        _increment("errors")
        _METRICS["last_error"] = f"Retrieval: {exc}"
        docs = []
    retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

    # ── Phase 4: Web fallback ──
    used_web_fallback = False
    if _should_fallback(docs):
        log.info("[routes] Low rerank scores — triggering Tavily fallback")
        _increment("web_fallbacks")
        used_web_fallback = True
        try:
            web_docs = await _web_fallback(retrieval_query, web_searcher)
            docs = web_docs or docs   # prefer web results; keep local if web fails
        except Exception as exc:
            log.warning("[routes] Web fallback failed: %s", exc)

    # ── Phase 5: Intent-aware generation (history-aware) ──
    sources = _format_sources(docs, include_content=include_content)
    t_generate = time.perf_counter()
    try:
        async for token in generator.generate(
            user_query=user_query,
            docs=docs,
            intent=intent_val,
            sub_entities=reflection.sub_entities,
            stream=stream,
            history_messages=history_messages,
        ):
            yield token
    except Exception as exc:
        log.error("[routes] Generation failed: %s", exc)
        _increment("errors")
        yield "\n\n[Lỗi hệ thống. Vui lòng thử lại.]"
    generate_ms = (time.perf_counter() - t_generate) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    log.info(
        "[routes] Done: intent=%s route=%s docs=%d web=%s "
        "reflect_ms=%.0f retrieve_ms=%.0f generate_ms=%.0f total_ms=%.0f",
        intent_val, route, len(docs), used_web_fallback,
        reflect_ms, retrieve_ms, generate_ms, total_ms,
    )

    # Meta event — always last in the stream.
    meta = json.dumps({"intent": intent_val, "sources": sources})
    yield f"__META__:{meta}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_fallback(docs: List[Dict]) -> bool:
    """
    Return True if the mean reranked score is below the fallback threshold,
    or if no docs were returned at all.
    """
    if not docs:
        return True
    scores = [d.get("score", 0.0) for d in docs]
    mean_score = sum(scores) / len(scores)
    return mean_score < _FALLBACK_SCORE_THRESHOLD


async def _web_fallback(query: str, searcher: Any) -> List[Dict]:
    """
    Invoke Tavily web search and normalise results to the pipeline doc shape:
        {cid, text, score, metadata: {source, title, from_web}}

    Calls search_trusted_laws() which already handles caching, circuit breaking,
    and singleflight dedup.  Returns an empty list on any failure.
    """
    try:
        web_docs = await searcher.search_trusted_laws(query)
        # web_docs are already plain dicts from the updated _parse_results;
        # they have the shape {cid, text, score, metadata}.
        return web_docs
    except Exception as exc:
        log.warning("[routes] _web_fallback error: %s", exc)
        return []


def _format_sources(docs: List[Dict], include_content: bool = False) -> List[Dict]:
    """
    Produce a compact list of source references to attach to the response.

    Strips the raw text (too large for the response object) and keeps only
    the metadata the frontend needs to render source chips, unless include_content is True.

    Supports both flat payloads (new parent-child pipeline: title/url at top level)
    and nested metadata dicts (legacy format).
    """
    sources = []
    for doc in docs:
        meta = doc.get("metadata", {})
        source_dict = {
            "id":     doc.get("cid") or doc.get("id", ""),
            "title":  meta.get("title") or doc.get("title", ""),
            "url":    meta.get("source") or meta.get("url") or doc.get("url", ""),
            "score":  round(doc.get("score", 0.0), 4),
            "entity": doc.get("entity_label", ""),
        }
        if include_content:
            source_dict["content"] = doc.get("text", "")
        sources.append(source_dict)
    return sources


def get_metrics() -> Dict[str, Any]:
    """Return a snapshot of the in-memory metrics dict."""
    return dict(_METRICS)