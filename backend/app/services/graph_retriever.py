"""
graph_retriever.py — GraphRAG local search via LanceDB.

Queries the LanceDB index built by Phase 6 (06_build_embeddings_lancedb.py)
to retrieve entity descriptions, community reports, and related text units
using vector similarity search.

Design decisions:
  - Reuses the DEk21 SentenceTransformer already loaded in main.py lifespan
    (zero extra init cost — same model, same 768-dim vectors).
  - LanceDB tables MUST be rebuilt with DEk21 embeddings (768-dim) before
    GraphRAG search works. Run 06_build_embeddings_lancedb.py with DEk21.
  - LanceDB queries are local file-based — no network round-trip.
  - Graph neighbor expansion uses in-memory entity→relationship lookup
    loaded from parquet at startup (fast, but ~400MB RAM for the full graph).
  - Returns results in the same {cid, text, score, metadata} format as
    the vector RAG retriever for seamless context merging.
  - Gracefully degrades: if LanceDB or parquet files are missing,
    returns empty results (caller falls back to vector RAG).

Tables queried:
  - entity_description: (id, name, description, vector) [768-dim DEk21]
  - community_full_content: (community, title, full_content, vector) [768-dim DEk21]

Optional (if loaded):
  - Relationships parquet for graph neighbor expansion.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("pipeline")

# Graph context is inherently noisier than precise text chunks, so we use
# a lower rerank threshold. With sigmoid scores (bimodal: ~0 or ~1), 0.15
# filters out clearly irrelevant results without being overly strict.
_GRAPH_RERANK_THRESHOLD = 0.15


class GraphRAGRetriever:
    """
    GraphRAG retriever using LanceDB for local vector search
    and optional graph neighbor expansion.

    Usage:
        retriever = GraphRAGRetriever(
            lancedb_path="path/to/lancedb",
            embedding_model=bge_m3_model,
        )
        await retriever.initialize()
        docs = await retriever.search("so sánh xe máy và ô tô", top_k=5)
    """

    def __init__(
        self,
        lancedb_path: str,
        embedding_model: Any = None,
        relationships_parquet: Optional[str] = None,
        local_reranker: Any = None,
    ):
        self._lancedb_path = lancedb_path
        self._embedder = embedding_model
        self._relationships_parquet = relationships_parquet
        self._local_reranker = local_reranker

        # Populated in initialize()
        self._db = None
        self._entity_table = None
        self._community_table = None
        self._rel_lookup: Dict[str, List[Dict]] = {}  # entity_name → [relationships]
        self._initialized = False

    async def initialize(self) -> None:
        """
        Open LanceDB connection and load relationship lookup.
        Called once during FastAPI lifespan startup.

        Skips initialization entirely if the LanceDB directory doesn't exist
        (e.g. GraphRAG pipeline hasn't been built yet).  This avoids loading
        the ~382 MB relationships parquet into ~1 GB of Python dicts for a
        feature that can't work without the search index.
        """
        if self._initialized:
            return

        try:
            import lancedb
            db_path = Path(self._lancedb_path)
            if not db_path.exists():
                log.warning(
                    "⚠️  GraphRAG disabled — LanceDB path not found: %s. "
                    "Skipping relationship loading to save ~1 GB RAM. "
                    "Run 06_build_embeddings_lancedb.py to enable GraphRAG.",
                    self._lancedb_path,
                )
                return

            self._db = lancedb.connect(str(db_path))
            tables = self._db.table_names()

            if "entity_description" in tables:
                self._entity_table = self._db.open_table("entity_description")
                log.info("✅ GraphRAG entity_description table loaded (%d rows)",
                         self._entity_table.count_rows())
            else:
                log.warning("[graph_retriever] entity_description table not found in LanceDB")

            if "community_full_content" in tables:
                self._community_table = self._db.open_table("community_full_content")
                log.info("✅ GraphRAG community_full_content table loaded (%d rows)",
                         self._community_table.count_rows())
            else:
                log.warning("[graph_retriever] community_full_content table not found in LanceDB")

            # Load relationship lookup for graph neighbor expansion
            # Only load if we have at least one searchable table.
            if self._entity_table or self._community_table:
                await self._load_relationships()

            self._initialized = True
            log.info("✅ GraphRAG retriever initialized.")

        except Exception as exc:
            log.error("❌ GraphRAG retriever init failed: %s", exc)

    async def _load_relationships(self) -> None:
        """
        Build an in-memory entity→relationships lookup from parquet.
        Used for graph neighbor expansion in COMPARE and MULTI_HOP.
        """
        if not self._relationships_parquet:
            return

        path = Path(self._relationships_parquet)
        if not path.exists():
            log.warning("[graph_retriever] relationships.parquet not found: %s", path)
            return

        try:
            import pandas as pd
            import time as _time
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                functools.partial(pd.read_parquet, str(path), columns=["source", "target", "description", "weight"]),
            )

            t0 = _time.monotonic()
            # Fill NaN and cast types once — avoids per-row overhead
            df["source"]      = df["source"].fillna("").astype(str).str.strip()
            df["target"]      = df["target"].fillna("").astype(str).str.strip()
            df["description"] = df["description"].fillna("").astype(str).str.strip()
            df["weight"]      = df["weight"].fillna(1.0).astype(float)

            # Keep only rows where both endpoints are non-empty
            df = df[(df["source"] != "") & (df["target"] != "")]

            # Build bidirectional lookup via records (much faster than iterrows)
            for row in df.itertuples(index=False):
                fwd = {"target": row.target, "description": row.description, "weight": row.weight}
                rev = {"target": row.source, "description": row.description, "weight": row.weight}
                self._rel_lookup.setdefault(row.source, []).append(fwd)
                self._rel_lookup.setdefault(row.target, []).append(rev)

            elapsed = _time.monotonic() - t0
            log.info("✅ Loaded %d entity relationship mappings in %.1fs", len(self._rel_lookup), elapsed)

        except Exception as exc:
            log.error("[graph_retriever] Failed to load relationships: %s", exc)

    # ------------------------------------------------------------------ #
    # Public search methods
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        top_k: int = 5,
        include_communities: bool = True,
        include_neighbors: bool = True,
    ) -> List[Dict]:
        """
        Search GraphRAG index for relevant entities and community reports.

        Parameters
        ----------
        query               : Search query string.
        top_k               : Number of top results per table.
        include_communities : Also search community reports.
        include_neighbors   : Expand results with graph neighbors.

        Returns
        -------
        List of docs in RAG-compatible format:
            [{cid, text, score, metadata: {source: "graphrag", type: ...}}]
        """
        if not self._initialized or not self._embedder:
            return []

        # Encode query with SentenceTransformer (DEk21, reuse existing model)
        try:
            query_vec = await self._encode_query(query)
        except Exception as exc:
            log.error("[graph_retriever] Query encoding failed: %s", exc)
            return []

        results: List[Dict] = []

        # 1. Search entities
        if self._entity_table:
            entity_docs = await self._search_table(
                self._entity_table, query_vec, top_k, doc_type="entity"
            )
            results.extend(entity_docs)

            # Expand with graph neighbors
            if include_neighbors and entity_docs:
                neighbor_docs = self._expand_neighbors(entity_docs, max_neighbors=3)
                results.extend(neighbor_docs)

        # 2. Search community reports
        if include_communities and self._community_table:
            community_docs = await self._search_table(
                self._community_table, query_vec, top_k, doc_type="community"
            )
            results.extend(community_docs)

        # Deduplicate by text content (same entity can appear in both tables)
        results = self._deduplicate(results)

        # Rerank with local reranker if available
        if self._local_reranker and results:
            try:
                results = await self._local_reranker.rerank(
                    query=query,
                    docs=results,
                    top_k=top_k * 2,
                )
                # Apply graph-specific threshold: drop clearly irrelevant results.
                # LocalReranker ranks but does not filter; we gate here.
                results = [d for d in results if d.get("score", 0) >= _GRAPH_RERANK_THRESHOLD]
            except Exception as exc:
                log.error("[graph_retriever] Local rerank failed: %s", exc)
                results.sort(key=lambda d: d.get("score", 0), reverse=True)
                results = results[:top_k * 2]
        else:
            # Sort by score descending
            results.sort(key=lambda d: d.get("score", 0), reverse=True)
            results = results[:top_k * 2]

        log.debug(
            "[graph_retriever] search: query=%.60r → %d results",
            query, len(results),
        )
        return results

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    async def _encode_query(self, query: str) -> list:
        """Encode query string into a 768-dim dense vector using SentenceTransformer (DEk21)."""
        try:
            from pyvi import ViTokenizer
            segmented_query = ViTokenizer.tokenize(query)
        except ImportError:
            log.warning("[graph_retriever] pyvi not installed, using unsegmented query (may degrade quality)")
            segmented_query = query

        loop = asyncio.get_event_loop()
        encode_fn = functools.partial(
            self._embedder.encode,
            [segmented_query],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        result = await loop.run_in_executor(None, encode_fn)
        return result[0].tolist()

    async def _search_table(
        self,
        table: Any,
        query_vec: list,
        top_k: int,
        doc_type: str,
    ) -> List[Dict]:
        """
        Vector similarity search on a LanceDB table.

        LanceDB search is synchronous and file-based (no network),
        so we run it in a thread executor.
        """
        loop = asyncio.get_event_loop()

        def _do_search():
            try:
                results = (
                    table.search(query_vec)
                    .limit(top_k)
                    .to_pandas()
                )
                return results
            except Exception as exc:
                log.error("[graph_retriever] LanceDB search failed (%s): %s", doc_type, exc)
                return None

        df = await loop.run_in_executor(None, _do_search)
        if df is None or df.empty:
            return []

        docs = []
        for _, row in df.iterrows():
            if doc_type == "entity":
                text = f"[Entity: {row.get('name', '')}]\n{row.get('description', '')}"
                cid = f"graphrag-entity-{row.get('id', '')}"
            elif doc_type == "community":
                text = f"[Community Report: {row.get('title', '')}]\n{row.get('full_content', '')}"
                cid = f"graphrag-community-{row.get('community', '')}"
            else:
                text = str(row.get("text", ""))
                cid = f"graphrag-{doc_type}-{row.get('id', '')}"

            # LanceDB returns _distance (L2) — convert to similarity score.
            # For cosine distance: similarity = 1 - distance
            distance = float(row.get("_distance", 1.0))
            score = max(0.0, 1.0 - distance)

            docs.append({
                "cid": cid,
                "id": cid,
                "text": text.strip(),
                "score": score,
                "metadata": {
                    "source": "graphrag",
                    "type": doc_type,
                },
            })

        return docs

    def _expand_neighbors(
        self,
        entity_docs: List[Dict],
        max_neighbors: int = 3,
    ) -> List[Dict]:
        """
        For each entity found, look up its graph neighbors (relationships)
        and add their descriptions as additional context.

        This is the key advantage of GraphRAG over vector RAG:
        it surfaces connected entities that vector search might miss.
        """
        if not self._rel_lookup:
            return []

        neighbor_docs = []
        seen_targets = set()

        for doc in entity_docs:
            # Extract entity name from the text "[Entity: <name>]\n..."
            text = doc.get("text", "")
            if text.startswith("[Entity: "):
                name_end = text.index("]")
                entity_name = text[len("[Entity: "):name_end]
            else:
                continue

            rels = self._rel_lookup.get(entity_name, [])
            # Sort by weight (strongest relationships first)
            rels_sorted = sorted(rels, key=lambda r: r.get("weight", 0), reverse=True)

            for rel in rels_sorted[:max_neighbors]:
                target = rel["target"]
                if target in seen_targets:
                    continue
                seen_targets.add(target)

                desc = rel.get("description", "")
                if not desc:
                    continue

                neighbor_docs.append({
                    "cid": f"graphrag-rel-{entity_name}-{target}",
                    "id": f"graphrag-rel-{entity_name}-{target}",
                    "text": f"[Relationship: {entity_name} → {target}]\n{desc}",
                    "score": doc.get("score", 0.5) * 0.8,  # Slightly lower than parent
                    "metadata": {
                        "source": "graphrag",
                        "type": "relationship",
                    },
                })

        return neighbor_docs

    def _deduplicate(self, docs: List[Dict]) -> List[Dict]:
        """Remove duplicate docs by cid, keeping the higher-scored one."""
        seen: Dict[str, Dict] = {}
        for doc in docs:
            cid = doc.get("cid", "")
            existing = seen.get(cid)
            if existing is None or doc.get("score", 0) > existing.get("score", 0):
                seen[cid] = doc
        return list(seen.values())
