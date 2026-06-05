#!/usr/bin/env python3
"""
06_build_embeddings_lancedb.py — Phase 6: BGE-M3 embeddings → LanceDB.

Reads:
  output/entities.parquet           → embeds 'description' → entity_description table
  output/community_reports.parquet  → embeds 'full_content' → community_full_content table

Writes:
  output/lancedb/  (vector_dim=1024, matches backend BGE-M3)

Checkpoint (row-level, survives crashes):
  output/phase6_checkpoint.json

Usage:
  python 06_build_embeddings_lancedb.py            # resume if checkpoint valid
  python 06_build_embeddings_lancedb.py --reset    # wipe and restart
  python 06_build_embeddings_lancedb.py --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.parent          # data_pipelines/graph_rag/
OUTPUT          = HERE / "output"
LANCEDB_PATH    = OUTPUT / "lancedb"
CHECKPOINT_PATH = OUTPUT / "phase6_checkpoint.json"
ENTITIES_PATH   = OUTPUT / "entities.parquet"
COMMUNITY_PATH  = OUTPUT / "community_reports.parquet"

VECTOR_DIM = 768   # DEk21 dense vector — must match backend

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase6")


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _load_cp() -> dict:
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception as e:
            log.warning("Checkpoint unreadable (%s) — starting fresh.", e)
    return {}


def _save_cp(cp: dict) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2))


def _cp_valid(cp: dict) -> bool:
    """Reject checkpoint if vector_dim doesn't match BGE-M3 1024."""
    for key in ("entity_description", "community_full_content"):
        dim = cp.get(key, {}).get("vector_dim")
        if dim and dim != VECTOR_DIM:
            log.warning(
                "Checkpoint vector_dim=%d for '%s' ≠ BGE-M3 %d — will reset.",
                dim, key, VECTOR_DIM,
            )
            return False
    return True


def _reset() -> None:
    log.info("🗑  Resetting checkpoint and LanceDB...")
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        log.info("   Deleted: %s", CHECKPOINT_PATH)
    if LANCEDB_PATH.exists():
        shutil.rmtree(LANCEDB_PATH)
        log.info("   Deleted: %s", LANCEDB_PATH)
    LANCEDB_PATH.mkdir(parents=True, exist_ok=True)


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model() -> Any:
    log.info("Loading DEk21 embedding model: huyydangg/DEk21_hcmute_embedding ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding")
    log.info("✅ DEk21 loaded successfully (dim=%d).", model.get_sentence_embedding_dimension())
    return model


def _embed(model: Any, texts: list[str]) -> list[list[float]]:
    """Encode texts → dense float32 vectors [N, 768]."""
    out = model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return out.tolist()


# ── LanceDB helpers ────────────────────────────────────────────────────────────

_ENTITY_SCHEMA = pa.schema([
    pa.field("id",          pa.string()),
    pa.field("name",        pa.string()),
    pa.field("description", pa.string()),
    pa.field("vector",      pa.list_(pa.float32(), VECTOR_DIM)),
])

_COMMUNITY_SCHEMA = pa.schema([
    pa.field("community",    pa.int64()),
    pa.field("title",        pa.string()),
    pa.field("full_content", pa.string()),
    pa.field("vector",       pa.list_(pa.float32(), VECTOR_DIM)),
])


def _open_or_create(db: Any, name: str, schema: pa.Schema) -> tuple[Any, int]:
    """Return (table, existing_row_count). Creates table if not present."""
    if name in db.table_names():
        tbl = db.open_table(name)
        n = tbl.count_rows()
        log.info("  ↩  Resuming '%s': %d rows already written.", name, n)
        return tbl, n
    tbl = db.create_table(name, schema=schema)
    return tbl, 0


# ── Embedding loops ────────────────────────────────────────────────────────────

def _run_table(
    *,
    model: Any,
    db: Any,
    table_key: str,
    df: pd.DataFrame,
    text_col: str,
    fallback_col: str,
    schema: pa.Schema,
    row_builder,          # callable(batch_df, vectors) → list[dict]
    batch_size: int,
    cp: dict,
) -> None:
    if cp.get(table_key, {}).get("status") == "completed":
        log.info("  ✅ '%s' already completed — skipping.", table_key)
        return

    total = len(df)
    log.info("  Rows to process: %d", total)

    tbl, skip = _open_or_create(db, table_key, schema)

    if skip >= total:
        log.info("  ✅ All rows in LanceDB — marking complete.")
        cp[table_key] = {"status": "completed", "completed_rows": total,
                         "total_rows": total, "vector_dim": VECTOR_DIM}
        _save_cp(cp)
        return

    remaining = df.iloc[skip:].reset_index(drop=True)
    log.info("  Starting from row %d/%d", skip, total)

    done = skip
    t0   = time.time()
    LOG_EVERY = max(1, 500 // batch_size)   # log every ~500 rows

    for i in range(0, len(remaining), batch_size):
        batch = remaining.iloc[i : i + batch_size]
        texts = batch[text_col].tolist()
        # Fallback to secondary column if primary is empty
        texts = [
            t if str(t).strip() else str(fb)
            for t, fb in zip(texts, batch[fallback_col].tolist())
        ]
        # Final fallback for fully empty entries
        texts = [t if str(t).strip() else "Không có mô tả" for t in texts]

        # Segment texts before encoding
        try:
            from pyvi import ViTokenizer
            segmented_texts = [ViTokenizer.tokenize(t) for t in texts]
        except ImportError:
            try:
                from underthesea import word_tokenize
                segmented_texts = [" ".join(word_tokenize(t)).replace(" ", "_") for t in texts]
            except ImportError:
                log.warning("Neither pyvi nor underthesea is installed. Using unsegmented texts.")
                segmented_texts = texts

        vectors = _embed(model, segmented_texts)
        rows    = row_builder(batch, vectors)
        tbl.add(rows)

        done += len(batch)
        cp[table_key] = {
            "status":         "in_progress",
            "completed_rows": done,
            "total_rows":     total,
            "vector_dim":     VECTOR_DIM,
        }
        _save_cp(cp)

        batch_idx = i // batch_size
        if batch_idx % LOG_EVERY == 0:
            elapsed = time.time() - t0
            rate    = (done - skip) / elapsed if elapsed > 0 else 0
            eta     = (total - done) / rate   if rate  > 0 else 0
            log.info(
                "  [%s] %d/%d (%.1f%%) | %.1f rows/s | ETA %.0fs",
                table_key, done, total, 100 * done / total, rate, eta,
            )

    cp[table_key] = {
        "status": "completed", "completed_rows": total,
        "total_rows": total, "vector_dim": VECTOR_DIM,
    }
    _save_cp(cp)
    log.info("  ✅ '%s' complete: %d rows.", table_key, total)


def embed_entities(model: Any, db: Any, batch_size: int, cp: dict) -> None:
    log.info("=" * 60)
    log.info("Phase 6A — entity_description")

    df = pd.read_parquet(ENTITIES_PATH, columns=["id", "name", "description"])
    df["id"]          = df["id"].fillna("").astype(str)
    df["name"]        = df["name"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    def _build_rows(batch: pd.DataFrame, vectors: list) -> list[dict]:
        return [
            {"id": r["id"], "name": r["name"],
             "description": r["description"], "vector": v}
            for (_, r), v in zip(batch.iterrows(), vectors)
        ]

    _run_table(
        model=model, db=db, table_key="entity_description",
        df=df, text_col="description", fallback_col="name",
        schema=_ENTITY_SCHEMA, row_builder=_build_rows,
        batch_size=batch_size, cp=cp,
    )


def embed_communities(model: Any, db: Any, batch_size: int, cp: dict) -> None:
    log.info("=" * 60)
    log.info("Phase 6B — community_full_content")

    df = pd.read_parquet(COMMUNITY_PATH, columns=["community", "title", "full_content"])
    df["community"]    = pd.to_numeric(df["community"], errors="coerce").fillna(0).astype("int64")
    df["title"]        = df["title"].fillna("").astype(str)
    df["full_content"] = df["full_content"].fillna("").astype(str)

    def _build_rows(batch: pd.DataFrame, vectors: list) -> list[dict]:
        return [
            {"community": int(r["community"]), "title": r["title"],
             "full_content": r["full_content"], "vector": v}
            for (_, r), v in zip(batch.iterrows(), vectors)
        ]

    _run_table(
        model=model, db=db, table_key="community_full_content",
        df=df, text_col="full_content", fallback_col="title",
        schema=_COMMUNITY_SCHEMA, row_builder=_build_rows,
        batch_size=batch_size, cp=cp,
    )


# ── Verification ───────────────────────────────────────────────────────────────

def verify(db: Any) -> None:
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: BGE-M3 embeddings → LanceDB")
    parser.add_argument("--reset",      action="store_true",
                        help="Delete checkpoint and LanceDB, start from scratch")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Embedding batch size (default: 32)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Phase 6: BGE-M3 Embeddings → LanceDB")
    log.info("  LanceDB : %s", LANCEDB_PATH)
    log.info("  Dim     : %d", VECTOR_DIM)
    log.info("  Batch   : %d", args.batch_size)
    log.info("=" * 60)

    # Validate input files
    for p in (ENTITIES_PATH, COMMUNITY_PATH):
        if not p.exists():
            log.error("❌ Required file not found: %s", p)
            sys.exit(1)

    # Checkpoint logic
    cp = _load_cp()
    if args.reset or not _cp_valid(cp):
        _reset()
        cp = {}
    else:
        log.info("✅ Valid checkpoint found — resuming.")
        LANCEDB_PATH.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model()

    # Connect LanceDB
    import lancedb
    db = lancedb.connect(str(LANCEDB_PATH))

    # Run
    embed_entities(model, db, args.batch_size, cp)
    embed_communities(model, db, args.batch_size, cp)

    # Verify
    verify(db)

    log.info("=" * 60)
    log.info("🎉 Phase 6 complete! LanceDB ready at: %s", LANCEDB_PATH)


if __name__ == "__main__":
    main()
