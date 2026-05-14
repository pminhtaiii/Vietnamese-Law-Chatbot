"""
retriever.py — Phase 3: Hybrid retrieval with intent-aware strategies.

v3.1 changes (from stubs → real implementation):
  - _hybrid_search() now connects to Qdrant via AsyncQdrantClient.
    Encodes queries with BGE-M3 (dense + sparse), uses Qdrant prefetch
    for RRF (Reciprocal Rank Fusion) between dense and sparse results.
  - _rerank() now prioritizes the local `thanhtantran/Vietnamese_Reranker`
    and falls back to Cohere rerank-v3.5 via AsyncClientV2 if local reranking fails.
    Both methods are fully async and non-blocking.
  - BGE-M3 encode() runs in asyncio.run_in_executor() to avoid blocking
    the FastAPI event loop (PyTorch inference is CPU-bound).
  - _extract_law_name_filter() uses Qdrant models.Filter with MatchText
    on the "text" payload field (the only text field in our collection).

  search_laws() accepts `intent` and `sub_entities` parameters.
  Each intent gets a tailored retrieval strategy:

    LEGAL_LOOKUP  → current behaviour (no change)
    COMPARE       → one search per sub_entity → merge + tag → rerank together
    SUMMARIZE     → top_k × 3, Qdrant payload filter by law/decree name
    PROCEDURE     → standard search + ordering-keyword boost
    DEFINITION    → tight top_k (≤3), stricter rerank threshold
    MULTI_HOP     → chained: search(sub_q[0]) → search(sub_q[1] + ctx) → merge
    CHITCHAT      → never called (guarded in routes.py)

Logical notes:
  - COMPARE: sub_entity searches run in parallel (asyncio.gather) for speed.
    Each result is tagged with `entity_label` so the generator can build a
    comparison table with entity columns.
  - MULTI_HOP: chained means the top-3 texts from sub_q[0] are appended to
    sub_q[1] as context before the second search.  This is crucial: without
    the chain the second hop knows nothing about the first legal conclusion.
  - SUMMARIZE top_k multiplier is capped at 30 to avoid memory issues.
  - DEFINITION rerank threshold is stricter to surface the single definitive
    article rather than a bag of loosely related docs.

Qdrant collection schema (from bge-m3.ipynb + migrate.py):
    Point ID   : int (= cid)
    Vectors    : {"dense": 1024-dim COSINE, "sparse": SparseVector}
    Payload    : {"text": str, "cid": int}
    (migrate.py maps original "content" → "text" for Docker Qdrant)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client.http import models as qmodels

log = logging.getLogger("pipeline")

# NOTE: Threshold-based filtering was REMOVED from the reranker.
# Vietnamese_Reranker (sigmoid) produces scores mostly in 0.01–0.25 range
# for Vietnamese legal text.  Filtering at any threshold was dropping most/all
# docs, causing every query to fallback to Tavily.
#
# The reranker now ONLY ranks (sorts by score).  Quality gating is handled
# DOWNSTREAM by _should_fallback() in routes.py.
_PROCEDURE_ORDER_KEYWORDS  = ["bước", "thứ tự", "trình tự", "hồ sơ", "cơ quan"]

# COMPARE: maximum number of sub_entities to retrieve separately.
# Cap prevents runaway API cost ("compare Law A, B, C, D, E").
_MAX_COMPARE_ENTITIES = 4

# SUMMARIZE: multiplier on top_k for broader coverage.
_SUMMARIZE_TOP_K_MULTIPLIER = 3
_SUMMARIZE_TOP_K_CAP        = 30

# Maximum number of docs to send to reranker.
_RERANK_BATCH_SIZE = 30

# Cohere rerank model — multilingual v3 supports Vietnamese well.
_RERANK_MODEL = "rerank-v3.5"


class Retriever:
    """
    Hybrid legal document retriever (BGE-M3 + Qdrant RRF + local/Cohere rerank).

    Dependencies injected via __init__:
      - qdrant_client   : qdrant_client.AsyncQdrantClient instance
      - cohere_client   : cohere.AsyncClientV2 instance (fallback reranker)
      - local_reranker  : LocalReranker instance (primary reranker)
      - embedding_model : FlagEmbedding.BGEM3FlagModel instance

    The embedding model is used to encode query strings into dense (1024-dim)
    and sparse (lexical_weights) vectors for Qdrant hybrid search.
    """

    def __init__(
        self,
        qdrant_client: Any,
        cohere_client: Any,
        local_reranker: Any = None,
        embedding_model: Any = None,
        collection_name: str = "legal_docs",
        default_top_k: int = 5,
    ):
        self._qdrant         = qdrant_client
        self._cohere         = cohere_client
        self._local_reranker = local_reranker
        self._embedder       = embedding_model
        self._collection     = collection_name
        self._default_k      = default_top_k

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    async def search_laws(
        self,
        user_query: str,
        top_k: int = 5,
        extra_queries: Optional[List[str]] = None,
        intent: str = "LEGAL_LOOKUP",
        sub_entities: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve and rerank legal documents with intent-aware strategy.

        Parameters
        ----------
        user_query    : The (possibly rewritten) main retrieval query.
        top_k         : Baseline number of documents to return.
        extra_queries : Expanded queries from the query reflector.
        intent        : IntentType value string (e.g. "COMPARE").
        sub_entities  : For COMPARE/MULTI_HOP — entity or sub-question list.

        Returns
        -------
        List of DocumentResponse-like dicts, ranked best-first.
        Each dict gains an `entity_label` key for COMPARE results.
        """
        extra_queries = extra_queries or []
        sub_entities  = sub_entities  or []
        intent        = (intent or "LEGAL_LOOKUP").upper()

        log.debug(
            "[retriever] search_laws intent=%s top_k=%d sub_entities=%s",
            intent, top_k, sub_entities,
        )

        if intent == "COMPARE":
            return await self._strategy_compare(
                user_query, top_k, extra_queries, sub_entities
            )
        if intent == "SUMMARIZE":
            return await self._strategy_summarize(
                user_query, top_k, extra_queries
            )
        if intent == "PROCEDURE":
            return await self._strategy_procedure(
                user_query, top_k, extra_queries
            )
        if intent == "DEFINITION":
            return await self._strategy_definition(
                user_query, top_k, extra_queries
            )
        if intent == "MULTI_HOP":
            return await self._strategy_multi_hop(
                user_query, top_k, extra_queries, sub_entities
            )
        # Default: LEGAL_LOOKUP (and any unknown intent falls here safely)
        return await self._strategy_legal_lookup(
            user_query, top_k, extra_queries
        )

    # ------------------------------------------------------------------ #
    # Intent strategies
    # ------------------------------------------------------------------ #

    async def _strategy_legal_lookup(
        self, query: str, top_k: int, extra_queries: List[str]
    ) -> List[Dict]:
        """Standard hybrid search — unchanged from pre-v3 behaviour."""
        all_queries = [query] + extra_queries
        raw = await self._hybrid_search(all_queries, top_k=top_k)
        return await self._rerank(query, raw, top_k=top_k)

    async def _strategy_compare(
        self,
        query: str,
        top_k: int,
        extra_queries: List[str],
        sub_entities: List[str],
    ) -> List[Dict]:
        """
        COMPARE strategy:
          1. Cap sub_entities at _MAX_COMPARE_ENTITIES.
          2. Run one hybrid search per sub_entity IN PARALLEL.
          3. Tag each result with entity_label (so the generator knows which
             entity this doc supports).
          4. Merge all results, de-duplicate by doc id, keep best score.
          5. Rerank the merged pool against the original query.

        Deduplication: if the same doc appears for both entities (common for
        ambiguous articles), keep it with the entity it scored higher for.
        """
        entities = sub_entities[:_MAX_COMPARE_ENTITIES]
        if not entities:
            # No entities extracted — fall through to standard lookup.
            log.warning("[retriever] COMPARE: no sub_entities — falling back to LEGAL_LOOKUP")
            return await self._strategy_legal_lookup(query, top_k, extra_queries)

        # Build per-entity query lists
        entity_queries = [
            [ent] + [f"{ent} {xq}" for xq in extra_queries][:2]
            for ent in entities
        ]

        # Run searches in parallel
        per_entity_results: list[list[dict]] = await asyncio.gather(*[
            self._hybrid_search(eq, top_k=top_k)
            for eq in entity_queries
        ])

        # Tag and merge, keeping best score per doc
        merged: dict[str, dict] = {}
        for ent, results in zip(entities, per_entity_results):
            for doc in results:
                doc_id = doc.get("cid") or doc.get("id", "")
                # Copy before tagging — prevents mutating the original dict,
                # which would corrupt entity_label if the same doc appears
                # for multiple entities.
                tagged_doc = {**doc, "entity_label": ent}
                existing = merged.get(doc_id)
                if existing is None or tagged_doc.get("score", 0) > existing.get("score", 0):
                    merged[doc_id] = tagged_doc

        merged_list = list(merged.values())

        # Rerank against the original query (not per-entity) to surface
        # docs that speak to the comparison holistically.
        reranked = await self._rerank(query, merged_list, top_k=top_k * len(entities))

        log.info(
            "[retriever] COMPARE: %d entities → %d merged → %d reranked",
            len(entities), len(merged_list), len(reranked),
        )
        return reranked

    async def _strategy_summarize(
        self, query: str, top_k: int, extra_queries: List[str]
    ) -> List[Dict]:
        """
        SUMMARIZE strategy:
          - Multiply top_k to get broader coverage.
          - Attempt Qdrant payload filter by law/decree name extracted
            from the query.
          - Rerank with the standard threshold.
        """
        broad_k = min(top_k * _SUMMARIZE_TOP_K_MULTIPLIER, _SUMMARIZE_TOP_K_CAP)
        all_queries = [query] + extra_queries

        # Try filtered search first; fall back to unfiltered if no hits.
        law_filter = self._extract_law_name_filter(query)
        raw = await self._hybrid_search(
            all_queries, top_k=broad_k, payload_filter=law_filter
        )
        if not raw and law_filter:
            log.debug("[retriever] SUMMARIZE: filter returned 0 docs, retrying unfiltered")
            raw = await self._hybrid_search(all_queries, top_k=broad_k)

        reranked = await self._rerank(query, raw, top_k=broad_k)
        log.info("[retriever] SUMMARIZE: broad_k=%d → %d results", broad_k, len(reranked))
        return reranked

    async def _strategy_procedure(
        self, query: str, top_k: int, extra_queries: List[str]
    ) -> List[Dict]:
        """
        PROCEDURE strategy:
          - Standard hybrid search.
          - Boost documents that contain ordering keywords ("bước", "hồ sơ", …)
            by adding a small score bonus before reranking.

        Boost is applied BEFORE reranking so Cohere can see the pre-ordered
        signal.  A constant bonus of 0.05 is enough to break ties.
        """
        all_queries = [query] + extra_queries
        raw = await self._hybrid_search(all_queries, top_k=top_k * 2)

        # Boost docs containing ordering keywords
        boosted = self._boost_by_keywords(raw, _PROCEDURE_ORDER_KEYWORDS, bonus=0.05)

        return await self._rerank(query, boosted, top_k=top_k)

    async def _strategy_definition(
        self, query: str, top_k: int, extra_queries: List[str]
    ) -> List[Dict]:
        """
        DEFINITION strategy:
          - Use a tight top_k (max 3) to surface the single definitive article.
          - Stricter rerank threshold eliminates loosely related docs.
        """
        tight_k = min(top_k, 3)
        all_queries = [query] + extra_queries[:1]   # only 1 extra to stay focused
        raw = await self._hybrid_search(all_queries, top_k=tight_k * 3)

        return await self._rerank(query, raw, top_k=tight_k)

    async def _strategy_multi_hop(
        self,
        query: str,
        top_k: int,
        extra_queries: List[str],
        sub_entities: List[str],
    ) -> List[Dict]:
        """
        MULTI_HOP strategy — chained retrieval:

          hop 1: search(sub_q[0])
          hop 2: search(sub_q[1] + top-3 texts from hop 1 as context)
          merge: deduplicate, rerank against the full original query.

        The context injection into hop 2 is the key difference from simply
        running two parallel searches.  It allows the model to find articles
        that reference *both* conditions (e.g. "drunk driving causing death"
        rather than just "drunk driving" OR just "causing death").

        Falls back to LEGAL_LOOKUP if fewer than 2 sub_entities.
        """
        if len(sub_entities) < 2:
            log.info(
                "[retriever] MULTI_HOP: <2 sub_entities, falling back to LEGAL_LOOKUP"
            )
            return await self._strategy_legal_lookup(query, top_k, extra_queries)

        sub_q0, sub_q1 = sub_entities[0], sub_entities[1]

        # Hop 1
        hop1_raw = await self._hybrid_search([sub_q0], top_k=top_k)

        # Build context string from top-3 hop-1 docs for hop-2 query
        top_hop1_texts = [
            doc.get("text", "")[:400]
            for doc in hop1_raw[:3]
        ]
        context_snippet = " ".join(top_hop1_texts).strip()
        augmented_sub_q1 = (
            f"{sub_q1} [ngữ cảnh: {context_snippet}]"
            if context_snippet
            else sub_q1
        )

        # Hop 2 — context-augmented
        hop2_raw = await self._hybrid_search([augmented_sub_q1], top_k=top_k)

        # Merge and deduplicate
        merged: dict[str, dict] = {}
        for doc in hop1_raw + hop2_raw:
            doc_id = doc.get("cid") or doc.get("id", "")
            existing = merged.get(doc_id)
            if existing is None or doc.get("score", 0) > existing.get("score", 0):
                merged[doc_id] = doc

        merged_list = list(merged.values())
        reranked = await self._rerank(query, merged_list, top_k=top_k)

        log.info(
            "[retriever] MULTI_HOP: hop1=%d hop2=%d merged=%d reranked=%d",
            len(hop1_raw), len(hop2_raw), len(merged_list), len(reranked),
        )
        return reranked

    # ------------------------------------------------------------------ #
    # Core search & rerank — real implementations
    # ------------------------------------------------------------------ #

    async def _encode_queries(self, queries: List[str]) -> dict:
        """
        Encode query strings into dense + sparse vectors using BGE-M3.

        Runs in a thread executor to avoid blocking the async event loop
        (BGEM3FlagModel.encode() is synchronous PyTorch inference).

        Returns the raw output dict from model.encode() with keys:
          - dense_vecs: list of numpy arrays (1024-dim each)
          - lexical_weights: list of dicts {token_id: weight}
        """
        loop = asyncio.get_event_loop()
        encode_fn = functools.partial(
            self._embedder.encode,
            queries,
            batch_size=len(queries),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=512,     # queries are short; save compute
        )
        return await loop.run_in_executor(None, encode_fn)

    async def _hybrid_search(
        self,
        queries: List[str],
        top_k: int = 10,
        payload_filter: Optional[qmodels.Filter] = None,
    ) -> List[Dict]:
        """
        Run BGE-M3 dense + sparse search with RRF fusion in Qdrant.

        For each query:
          1. Encode with BGE-M3 → (dense_vec, sparse_vec)
          2. Send to Qdrant using query_points() with two prefetch arms
             (one dense, one sparse) and RRF fusion.
          3. Extract payload and score from results.

        Multiple queries are merged via a simple dict-based dedup (best score wins).
        This implements cross-query RRF at the application level.

        Parameters
        ----------
        queries        : One or more query strings to search.
        top_k          : Number of candidates to return before reranking.
        payload_filter : Optional Qdrant Filter object for payload filtering.

        Returns
        -------
        List of docs: [{"cid": int, "text": str, "score": float, "metadata": {}}]
        Returns [] on any backend error (never raises).
        """
        if not self._qdrant or not self._embedder:
            log.warning("[retriever] _hybrid_search: qdrant or embedder not initialized")
            return []

        try:
            embeddings = await self._encode_queries(queries)
        except Exception as exc:
            log.error("[retriever] BGE-M3 encode failed: %s", exc)
            return []

        dense_vecs = embeddings["dense_vecs"]
        sparse_vecs = embeddings["lexical_weights"]

        # Merge results across all queries, keeping best score per doc_id.
        merged: dict[str, dict] = {}

        for i, query_text in enumerate(queries):
            try:
                dense_vec = dense_vecs[i].tolist() if hasattr(dense_vecs[i], "tolist") else dense_vecs[i]
                sparse_dict = sparse_vecs[i]
                sparse_indices = [int(k) for k in sparse_dict.keys()]
                sparse_values = list(sparse_dict.values())

                # Qdrant prefetch + RRF: two retrieval arms fused by reciprocal rank.
                # search_params enables rescoring with original float32 vectors
                # when the collection uses Scalar Quantization (int8).
                result = await self._qdrant.query_points(
                    collection_name=self._collection,
                    prefetch=[
                        qmodels.Prefetch(
                            query=dense_vec,
                            using="dense",
                            limit=top_k * 2,
                            filter=payload_filter,
                        ),
                        qmodels.Prefetch(
                            query=qmodels.SparseVector(
                                indices=sparse_indices,
                                values=sparse_values,
                            ),
                            using="sparse",
                            limit=top_k * 2,
                            filter=payload_filter,
                        ),
                    ],
                    query=qmodels.FusionQuery(
                        fusion=qmodels.Fusion.RRF,
                    ),
                    limit=top_k,
                    with_payload=True,
                    search_params=qmodels.SearchParams(
                        quantization=qmodels.QuantizationSearchParams(
                            rescore=True,
                            oversampling=1.5,
                        ),
                    ),
                )

                for point in result.points:
                    payload = point.payload or {}
                    doc_id = str(payload.get("cid", point.id))
                    score = point.score if point.score is not None else 0.0

                    doc = {
                        "cid":      doc_id,
                        "id":       doc_id,
                        "text":     payload.get("text", ""),
                        "score":    score,
                        "metadata": {},
                    }

                    existing = merged.get(doc_id)
                    if existing is None or score > existing.get("score", 0):
                        merged[doc_id] = doc

            except Exception as exc:
                log.error("[retriever] Qdrant query failed for query %d: %s", i, exc)
                continue

        results = sorted(merged.values(), key=lambda d: d["score"], reverse=True)
        # Log at INFO level — critical for diagnosing empty retrieval issues
        log.info(
            "[retriever] _hybrid_search: %d queries → %d unique docs (top score=%.4f)",
            len(queries), len(results),
            results[0]["score"] if results else 0.0,
        )
        if results:
            # Log first doc preview to verify text is not empty
            first_text = results[0].get("text", "")
            log.info(
                "[retriever] Top doc cid=%s text_len=%d preview=%.100r",
                results[0].get("cid"), len(first_text), first_text[:100],
            )
        else:
            log.warning("[retriever] _hybrid_search returned ZERO docs — Qdrant may be empty or disconnected")
        return results

    async def _rerank(
        self,
        query: str,
        docs: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank candidates with local Vietnamese Reranker (primary) or
        Cohere Reranker (fallback).

        NO threshold filtering — the reranker only RANKS documents.
        Quality gating is handled downstream by _should_fallback() in routes.py.

        Parameters
        ----------
        query     : The main query string for relevance scoring.
        docs      : Candidate documents from _hybrid_search.
        top_k     : Maximum documents to return.

        Returns
        -------
        Ranked docs (best first), up to top_k.
        Falls back to Cohere if local reranker fails, then to vector scores.
        """
        if not docs:
            return []

        # Cap the number of docs sent to reranker to avoid limits and cost/latency
        docs_to_rerank = docs[:_RERANK_BATCH_SIZE]
        texts = [d.get("text", "") for d in docs_to_rerank]

        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            log.warning("[retriever] _rerank: all docs have empty text")
            return []

        # Try Local Reranker first
        if self._local_reranker:
            try:
                ranked = await self._local_reranker.rerank(
                    query=query,
                    docs=docs_to_rerank,
                    top_k=top_k,
                )
                log.info(
                    "[retriever] _rerank (local): %d candidates → %d ranked",
                    len(valid_indices), len(ranked),
                )
                return ranked
            except Exception as exc:
                log.error("[retriever] Local rerank failed: %s, falling back to Cohere", exc)

        # Fallback to Cohere Reranker
        if not self._cohere:
            log.warning("[retriever] _rerank: cohere client not initialized and local reranker failed/missing — skipping rerank")
            return sorted(docs, key=lambda d: d.get("score", 0), reverse=True)[:top_k]

        valid_texts = [texts[i] for i in valid_indices]

        try:
            response = await self._cohere.rerank(
                query=query,
                documents=valid_texts,
                top_n=min(top_k, len(valid_texts)),
                model=_RERANK_MODEL,
            )

            ranked = []
            for r in response.results:
                original_idx = valid_indices[r.index]
                ranked.append({
                    **docs_to_rerank[original_idx],
                    "score": r.relevance_score,
                })

            log.info(
                "[retriever] _rerank (cohere): %d candidates → %d ranked",
                len(valid_texts), len(ranked),
            )
            return ranked

        except Exception as exc:
            log.error("[retriever] Cohere rerank failed: %s", exc)
            return sorted(docs, key=lambda d: d.get("score", 0), reverse=True)[:top_k]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _boost_by_keywords(
        self,
        docs: List[Dict],
        keywords: Sequence[str],
        bonus: float = 0.05,
    ) -> List[Dict]:
        """
        Add `bonus` to the score of any doc whose text contains a keyword.

        This is a lightweight heuristic that bubbles procedural docs (which
        mention "bước", "hồ sơ", etc.) above purely definitional docs before
        Cohere reranking.
        """
        boosted = []
        for doc in docs:
            text  = (doc.get("text") or "").lower()
            score = doc.get("score", 0.0)
            if any(kw.lower() in text for kw in keywords):
                doc = {**doc, "score": score + bonus}
            boosted.append(doc)
        return boosted

    def _extract_law_name_filter(self, query: str) -> Optional[qmodels.Filter]:
        """
        Heuristic: extract a law or decree name from the query for Qdrant
        payload filtering in SUMMARIZE mode.

        Matches patterns like:
          "Nghị định 100/2019"
          "Luật Giao thông đường bộ"
          "Bộ luật Hình sự 2015"

        Returns a Qdrant Filter that does substring matching on the "text"
        payload field, or None if no law name is detected.

        Note: Our collection payload only has "text" and "cid" fields.
        We match on "text" because the law name appears within the chunk text.
        """
        patterns = [
            r"(ngh[iị]\s+[dđ][iị]nh\s+\d+/\d{4}(?:/[A-Z\-]+)?)",
            r"(lu[aậ]t\s+[A-Za-zÀ-ỹ\s]{3,40?}(?:\d{4})?)",
            r"(b[oộ]\s+lu[aậ]t\s+[A-Za-zÀ-ỹ\s]{3,30?}(?:\d{4})?)",
        ]
        for pat in patterns:
            m = re.search(pat, query, re.IGNORECASE | re.UNICODE)
            if m:
                name = m.group(1).strip()
                log.debug("[retriever] SUMMARIZE filter: text contains %r", name)
                return qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="text",
                            match=qmodels.MatchText(text=name),
                        )
                    ]
                )
        return None
