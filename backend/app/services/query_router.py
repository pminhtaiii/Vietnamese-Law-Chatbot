"""
query_router.py — Intent-based routing between Vector RAG and GraphRAG.

Maps each intent from the 7-intent taxonomy to a retrieval strategy:
  - "rag"      → Vector RAG only (Qdrant + Cohere rerank)
  - "graphrag" → GraphRAG only (LanceDB local search)
  - "both"     → RAG + GraphRAG in parallel (asyncio.gather)
  - "none"     → No retrieval (CHITCHAT short-circuit)

Design principles:
  1. Zero latency overhead: routing is a dict lookup on the intent already
     computed by QueryReflector — no extra LLM call.
  2. Simple: ~30 lines of logic. No ML classifier, no feature engineering.
  3. Configurable: ROUTE_MAP can be overridden, but defaults are hardcoded
     for simplicity per CLAUDE.md guidelines.

Routing rationale:
  - LEGAL_LOOKUP / DEFINITION / PROCEDURE → RAG: these need precise text
    chunk retrieval; vector search + rerank excels here.
  - COMPARE / MULTI_HOP → GraphRAG: these need entity relationships and
    cross-reference reasoning; graph traversal is essential.
  - SUMMARIZE → Both: community reports give global overview (GraphRAG),
    while text chunks give specific details (RAG).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.services.query_reflector import IntentType

log = logging.getLogger("pipeline")

# Route definitions — simple, explicit, no magic.
ROUTE_MAP: Dict[IntentType, str] = {
    IntentType.LEGAL_LOOKUP: "rag",
    IntentType.DEFINITION:   "rag",
    IntentType.PROCEDURE:    "rag",
    IntentType.COMPARE:      "rag",   # TODO: enable "both" after entity grounding bridge is built
    IntentType.MULTI_HOP:    "rag",   # TODO: enable "both" after entity grounding bridge is built
    IntentType.SUMMARIZE:    "both",  # Community reports (GraphRAG) + text chunks (RAG) in parallel
    IntentType.CHITCHAT:     "none",
}


def get_route(intent: IntentType) -> str:
    """
    Return the retrieval strategy for a given intent.

    Returns "rag" as safe default for unknown intents.
    """
    return ROUTE_MAP.get(intent, "rag")


async def route_retrieval(
    query: str,
    intent: IntentType,
    rag_retriever: Any,
    graph_retriever: Any,
    top_k: int = 5,
    extra_queries: Optional[List[str]] = None,
    intent_val: str = "LEGAL_LOOKUP",
    sub_entities: Optional[List[str]] = None,
    local_reranker: Any = None,
) -> List[Dict]:
    """
    Execute retrieval through the appropriate backend(s) based on intent.

    Parameters
    ----------
    query           : The refined query string for retrieval.
    intent          : IntentType enum value from QueryReflector.
    rag_retriever   : The existing Retriever instance (Qdrant + Cohere).
    graph_retriever : The GraphRAGRetriever instance (LanceDB).
    top_k           : Number of documents to retrieve.
    extra_queries   : Expanded queries from QueryReflector.
    intent_val      : String value of intent (for Retriever.search_laws).
    sub_entities    : Sub-entities for COMPARE/MULTI_HOP.

    Returns
    -------
    List of documents in unified format [{cid, text, score, metadata}].
    """
    route = get_route(intent)

    if route == "none":
        return []

    if route == "rag":
        return await _rag_search(
            rag_retriever, query, top_k, extra_queries, intent_val, sub_entities
        )

    if route == "graphrag":
        # Try GraphRAG first; if it returns nothing, fall back to RAG.
        graph_docs = await _graphrag_search(graph_retriever, query, top_k)
        if graph_docs:
            return graph_docs
        log.info("[router] GraphRAG returned 0 results — falling back to RAG")
        return await _rag_search(
            rag_retriever, query, top_k, extra_queries, intent_val, sub_entities
        )

    if route == "both":
        # Run RAG and GraphRAG in parallel for minimum latency.
        rag_task = _rag_search(
            rag_retriever, query, top_k, extra_queries, intent_val, sub_entities
        )
        graph_task = _graphrag_search(graph_retriever, query, top_k)

        rag_docs, graph_docs = await asyncio.gather(
            rag_task, graph_task, return_exceptions=True
        )

        # Handle exceptions from either side gracefully.
        if isinstance(rag_docs, Exception):
            log.error("[router] RAG search failed in 'both' mode: %s", rag_docs)
            rag_docs = []
        if isinstance(graph_docs, Exception):
            log.error("[router] GraphRAG search failed in 'both' mode: %s", graph_docs)
            graph_docs = []

        # Both RAG and GraphRAG already rerank internally — just merge by score.
        # No second rerank call: avoids doubling CPU + latency.
        merged_docs = _merge_results(rag_docs, graph_docs, top_k * 3)
        return merged_docs[:top_k * 2]

    # Unknown route — safe default.
    return await _rag_search(
        rag_retriever, query, top_k, extra_queries, intent_val, sub_entities
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _rag_search(
    retriever: Any,
    query: str,
    top_k: int,
    extra_queries: Optional[List[str]],
    intent_val: str,
    sub_entities: Optional[List[str]],
) -> List[Dict]:
    """Delegate to the existing vector RAG retriever."""
    try:
        return await retriever.search_laws(
            user_query=query,
            top_k=top_k,
            extra_queries=extra_queries,
            intent=intent_val,
            sub_entities=sub_entities,
        )
    except Exception as exc:
        log.error("[router] RAG search failed: %s", exc)
        return []


async def _graphrag_search(
    graph_retriever: Any,
    query: str,
    top_k: int,
) -> List[Dict]:
    """Delegate to the GraphRAG retriever."""
    if graph_retriever is None:
        return []
    try:
        return await graph_retriever.search(query=query, top_k=top_k)
    except Exception as exc:
        log.error("[router] GraphRAG search failed: %s", exc)
        return []


def _merge_results(
    rag_docs: List[Dict],
    graph_docs: List[Dict],
    max_docs: int = 10,
) -> List[Dict]:
    """
    Merge results from RAG and GraphRAG, deduplicating by cid.

    For the "both" route: interleave results to give balanced coverage.
    When cids collide (unlikely between RAG/GraphRAG), keep the higher score.
    """
    merged: Dict[str, Dict] = {}

    for doc in rag_docs + graph_docs:
        cid = doc.get("cid") or doc.get("id", "")
        existing = merged.get(cid)
        if existing is None or doc.get("score", 0) > existing.get("score", 0):
            merged[cid] = doc

    # Sort by score descending and cap at max_docs.
    result = sorted(merged.values(), key=lambda d: d.get("score", 0), reverse=True)
    return result[:max_docs]
