#!/usr/bin/env python3
"""
07_convert_sqlite_to_lancedb.py — Phase 7: SQLite → LanceDB converter.

Reads the pre-computed 768-dim DEk21 vectors (BLOBs) from phase6.sqlite
(produced by Phase 6 on Kaggle) and writes them directly into LanceDB
.lance tables — the format expected by GraphRAGRetriever in the backend.

No re-embedding is done here. Phase 6 already embedded with DEk21 (768-dim).

Input  (place the file here before running):
    data_pipelines/graph_rag/output/lancedb/phase6.sqlite

Output (LanceDB tables, read-only mounted into backend Docker):
    data_pipelines/graph_rag/output/lancedb/
        entity_description.lance/
        community_full_content.lance/

Usage:
    python scripts/07_convert_sqlite_to_lancedb.py
    python scripts/07_convert_sqlite_to_lancedb.py --reset        # overwrite existing tables
    python scripts/07_convert_sqlite_to_lancedb.py --batch-size 2000
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Generator

import pyarrow as pa

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent.parent                 # data_pipelines/graph_rag/
LANCEDB_DIR = HERE / "output" / "lancedb"
SQLITE_PATH = LANCEDB_DIR / "phase6.sqlite"

VECTOR_DIM = 768   # DEk21 dense output — must match backend settings

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sqlite2lancedb")


# ── PyArrow schemas (must match GraphRAGRetriever expectations) ────────────────

ENTITY_SCHEMA = pa.schema([
    pa.field("id",          pa.string()),
    pa.field("name",        pa.string()),
    pa.field("description", pa.string()),
    pa.field("vector",      pa.list_(pa.float32(), VECTOR_DIM)),
])

COMMUNITY_SCHEMA = pa.schema([
    pa.field("community",    pa.int64()),
    pa.field("title",        pa.string()),
    pa.field("full_content", pa.string()),
    pa.field("vector",       pa.list_(pa.float32(), VECTOR_DIM)),
])


# ── BLOB unpacker ──────────────────────────────────────────────────────────────

def _unpack(blob: bytes) -> list[float]:
    """Unpack a raw float32 BLOB written by phase6's _pack_vector()."""
    return list(struct.unpack(f"{VECTOR_DIM}f", blob))


# ── Batch iterators ────────────────────────────────────────────────────────────

def _iter_entity_batches(
    conn: sqlite3.Connection, batch_size: int
) -> Generator[pa.RecordBatch, None, None]:
    cur = conn.execute(
        "SELECT id, name, description, vector FROM entity_description ORDER BY row_num"
    )
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        ids, names, descs, vecs = [], [], [], []
        for entity_id, name, desc, blob in rows:
            ids.append(entity_id or "")
            names.append(name or "")
            descs.append(desc or "")
            vecs.append(_unpack(blob))
        yield pa.record_batch([ids, names, descs, vecs], schema=ENTITY_SCHEMA)


def _iter_community_batches(
    conn: sqlite3.Connection, batch_size: int
) -> Generator[pa.RecordBatch, None, None]:
    cur = conn.execute(
        "SELECT community, title, full_content, vector FROM community_full_content ORDER BY row_num"
    )
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        communities, titles, contents, vecs = [], [], [], []
        for community_id, title, full_content, blob in rows:
            communities.append(int(community_id or 0))
            titles.append(title or "")
            contents.append(full_content or "")
            vecs.append(_unpack(blob))
        yield pa.record_batch([communities, titles, contents, vecs], schema=COMMUNITY_SCHEMA)


# ── LanceDB write helper ───────────────────────────────────────────────────────

def _write_table(db, table_name: str, schema: pa.Schema, batch_iter, reset: bool) -> int:
    if table_name in db.table_names():
        if reset:
            log.info("  --reset: dropping existing '%s'", table_name)
            db.drop_table(table_name)
        else:
            n = db.open_table(table_name).count_rows()
            log.info("  '%s' already exists (%d rows). Use --reset to overwrite.", table_name, n)
            return n

    tbl = None
    total = 0
    t0 = time.time()

    for batch in batch_iter:
        if tbl is None:
            tbl = db.create_table(table_name, data=batch, schema=schema)
        else:
            tbl.add(batch)
        total += len(batch)
        log.info("  '%s': wrote %d rows so far...", table_name, total)

    elapsed = time.time() - t0
    log.info("  ✅ '%s' done: %d rows in %.1fs", table_name, total, elapsed)
    return total


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7: Copy DEk21 vectors from phase6.sqlite → LanceDB .lance tables"
    )
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate LanceDB tables if they already exist")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Rows per batch (default: 2000)")
    parser.add_argument("--sqlite", type=str, default=str(SQLITE_PATH),
                        help=f"Path to phase6.sqlite (default: {SQLITE_PATH})")
    parser.add_argument("--lancedb-dir", type=str, default=str(LANCEDB_DIR),
                        help=f"LanceDB output directory (default: {LANCEDB_DIR})")

    try:
        get_ipython  # type: ignore[name-defined]  # noqa: F821
        _in_notebook = True
    except NameError:
        _in_notebook = False
    args = parser.parse_args(args=[] if _in_notebook else None)

    sqlite_path = Path(args.sqlite)
    lancedb_dir = Path(args.lancedb_dir)

    log.info("=" * 60)
    log.info("Phase 7: SQLite → LanceDB (BLOB copy, no re-embed)")
    log.info("  SQLite  : %s", sqlite_path)
    log.info("  LanceDB : %s", lancedb_dir)
    log.info("  Dim     : %d", VECTOR_DIM)
    log.info("  Batch   : %d", args.batch_size)
    log.info("  Reset   : %s", args.reset)
    log.info("=" * 60)

    if not sqlite_path.exists():
        log.error("❌ SQLite file not found: %s\n   Place phase6.sqlite in: %s",
                  sqlite_path, lancedb_dir)
        sys.exit(1)

    lancedb_dir.mkdir(parents=True, exist_ok=True)

    try:
        import lancedb
    except ImportError:
        log.error("❌ lancedb not installed. Run: pip install lancedb pyarrow")
        sys.exit(1)

    conn = sqlite3.connect(str(sqlite_path))
    db   = lancedb.connect(str(lancedb_dir))
    log.info("Connected to SQLite and LanceDB.")

    tables_in_sqlite = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    for required in ("entity_description", "community_full_content"):
        if required not in tables_in_sqlite:
            log.error("❌ Table '%s' not found in SQLite. Was Phase 6 completed?", required)
            conn.close()
            sys.exit(1)

    log.info("=" * 60)
    log.info("Writing entity_description...")
    entity_rows = _write_table(
        db, "entity_description", ENTITY_SCHEMA,
        _iter_entity_batches(conn, args.batch_size), args.reset,
    )

    log.info("=" * 60)
    log.info("Writing community_full_content...")
    community_rows = _write_table(
        db, "community_full_content", COMMUNITY_SCHEMA,
        _iter_community_batches(conn, args.batch_size), args.reset,
    )

    conn.close()

    log.info("=" * 60)
    log.info("Verification:")
    all_ok = True
    for name in ("entity_description", "community_full_content"):
        if name in db.table_names():
            n = db.open_table(name).count_rows()
            log.info("  ✅ %-30s : %d rows", name, n)
        else:
            log.error("  ❌ %-30s : TABLE NOT FOUND", name)
            all_ok = False

    if not all_ok:
        sys.exit(1)

    log.info("=" * 60)
    log.info("🎉 Done! LanceDB ready at: %s", lancedb_dir)
    log.info("   entity_description    : %d rows", entity_rows)
    log.info("   community_full_content: %d rows", community_rows)
    log.info("   Backend can now serve GraphRAG queries.")


if __name__ == "__main__":
    main()
