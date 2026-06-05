"""
parent_store.py — Parent text lookup from parents.sqlite.

The pipeline splits parent and child data into separate SQLite files:
  - parents.sqlite  : parent_id → parent_text + metadata (one row per parent)
  - children_*.sqlite: cid → child text + vectors (one row per child, pushed to Qdrant)

Qdrant stores only lean child payloads (cid, parent_id, text, doc_id).
At query time, the retriever hits Qdrant for child matches, then uses this
ParentStore to batch-resolve parent_id → parent_text for the LLM context.

Usage:
    parent_store = ParentStore("data/rag_sqlite/parents.sqlite")
    texts = parent_store.get_parent_texts({12345, 67890})
    # → {12345: "Điều 12. ...\n\nĐiều 13. ...", 67890: "..."}

    meta = parent_store.get_parent_metadata({12345})
    # → {12345: {"title": "...", "document_number": "...", ...}}
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Set

log = logging.getLogger("pipeline")


class ParentStore:
    """
    Read-only store for parent chunk text and metadata.

    Opens the SQLite file once on init with immutable mode for safe
    concurrent reads.  All lookups are indexed by parent_id (PRIMARY KEY).
    """

    def __init__(self, db_path: str):
        self._path = Path(db_path)
        if not self._path.exists():
            raise FileNotFoundError(f"parents.sqlite not found: {self._path}")

        # Open in read-only immutable mode for safe concurrent access
        uri = f"file:{self._path}?mode=ro&immutable=1"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Verify table exists and count rows
        try:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM parents").fetchone()
            count = row["n"] if row else 0
            log.info(
                "ParentStore loaded: %s (%d parents)", self._path, count
            )
        except sqlite3.OperationalError:
            # Table doesn't exist — maybe empty DB or wrong schema
            log.warning("ParentStore: 'parents' table not found in %s", self._path)

    def get_parent_texts(self, parent_ids: Set[int]) -> Dict[int, str]:
        """
        Batch-fetch parent_text for a set of parent_ids.

        Returns a dict mapping parent_id → parent_text.
        Missing IDs are silently omitted.
        """
        if not parent_ids:
            return {}
        ids = list(parent_ids)
        # Chunk into groups of 999 (SQLite parameter limit)
        out: Dict[int, str] = {}
        for i in range(0, len(ids), 999):
            batch = ids[i : i + 999]
            placeholders = ",".join("?" for _ in batch)
            sql = f"SELECT parent_id, parent_text FROM parents WHERE parent_id IN ({placeholders})"
            try:
                for row in self._conn.execute(sql, batch):
                    out[int(row["parent_id"])] = row["parent_text"] or ""
            except sqlite3.OperationalError as exc:
                log.error("ParentStore.get_parent_texts query error: %s", exc)
        return out

    def get_parent_metadata(self, parent_ids: Set[int]) -> Dict[int, Dict[str, Any]]:
        """
        Batch-fetch metadata for a set of parent_ids.

        Returns a dict mapping parent_id → metadata dict.
        """
        if not parent_ids:
            return {}
        ids = list(parent_ids)
        out: Dict[int, Dict[str, Any]] = {}
        meta_cols = [
            "parent_id", "doc_id", "document_number", "title",
            "legal_type", "legal_sectors", "issuing_authority",
            "issuance_date", "url",
        ]
        col_str = ", ".join(meta_cols)
        for i in range(0, len(ids), 999):
            batch = ids[i : i + 999]
            placeholders = ",".join("?" for _ in batch)
            sql = f"SELECT {col_str} FROM parents WHERE parent_id IN ({placeholders})"
            try:
                for row in self._conn.execute(sql, batch):
                    meta = {col: row[col] for col in meta_cols}
                    out[int(row["parent_id"])] = meta
            except sqlite3.OperationalError as exc:
                log.error("ParentStore.get_parent_metadata query error: %s", exc)
        return out

    def close(self) -> None:
        if self._conn:
            self._conn.close()