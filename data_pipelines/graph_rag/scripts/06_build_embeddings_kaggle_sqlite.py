#!/usr/bin/env python3
"""
06_build_embeddings_kaggle_sqlite.py - Phase 6: DEk21 embeddings -> SQLite.

Reads:
  output/entities.parquet           -> embeds 'description' -> entity_description table
  output/community_reports.parquet  -> embeds 'full_content' -> community_full_content table

Writes:
  output/sqlite_outputs/phase6.sqlite (768-dim DEk21 vectors)

Checkpoint:
  output/phase6_sqlite_checkpoint.json

Usage:
  python 06_build_embeddings_kaggle_sqlite.py
  python 06_build_embeddings_kaggle_sqlite.py --reset
  python 06_build_embeddings_kaggle_sqlite.py --batch-size 32

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY NO text_units.parquet?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
text_units.parquet is intentionally EXCLUDED. Trade-off analysis:

BENEFIT of embedding text_units:
  - Enables direct chunk-level retrieval from the GraphRAG pipeline,
    bypassing the entity/community hop. Slightly higher recall on
    narrow, verbatim-factual queries.

COST (why we skip it):
  - File size: text_units.parquet is typically 5–20× larger than
    entities.parquet for the same corpus (raw legal text per chunk).
  - ~153K entities embed in ~40 min on CPU; text_units can be
    400K–1M rows, pushing runtime to 4–10 h on CPU.
  - Storage: at 1024 floats × 4 bytes/float, 1M rows = ~4 GB of
    binary vectors in SQLite, straining the 2 GB default page cache.
  - The backend already has dense chunk retrieval from Qdrant
    (primary vector store). GraphRAG SQLite is used only for the
    graph-aware entity/community path (GraphRAGRetriever). Adding
    text_units would duplicate Qdrant's job with inferior tooling.

VERDICT: skip text_units. If you need raw-chunk GraphRAG retrieval,
re-index those chunks back into Qdrant instead of SQLite.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPENDENCY NOTE (why we bootstrap inside the script):
  The previous approach relied on requirements_phase6.txt which pins
  transformers==4.44.2 and torch==2.5.1+cpu. FlagEmbedding>=1.3
  requires transformers>=4.45.0, causing a conflict that the
  patch_transformers_compatibility() shim tried (imperfectly) to paper
  over. Following the Kaggle notebook pattern, we install a resolved
  dependency set at runtime before any heavy imports, so the process
  always starts with a consistent environment regardless of what the
  host already has installed. [ref: FlagEmbedding changelog v1.3,
  huggingface/transformers release notes 4.45]
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Bootstrap dependencies (notebook-style, runs before any ML imports)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
import sys

# ── Batch 1: ML core — mirrors Kaggle cell 1 ──────────────────────────────
# Intentionally does NOT include transformers or torch.
#
# Why NOT transformers:
#   Kaggle pre-installs a coherent transformers build (all .py files in sync).
#   pip-installing a different version WHILE the kernel is running causes a
#   partial overwrite: some .py files (e.g. quantizers/quantizer_quanto.py)
#   stay from the old version while utils/__init__.py is replaced by the new,
#   producing:
#     ImportError: cannot import name 'is_quanto_available' from transformers.utils
#   FlagEmbedding 1.2.10 only requires transformers>=4.30 and works fine
#   with whatever Kaggle has pre-installed — leave it untouched.
#
# Why NOT torch/torchvision:
#   Kaggle pre-installs a matched torch + torchvision + CUDA triple.
#   Upgrading torch alone breaks torchvision ABI (nms operator missing).
_PACKAGES_ML = [
    "transformers<4.45.0",
    "huggingface_hub>=0.23.2,<1.0",
    "sentence-transformers",
    "pyvi",
    "qdrant-client",
]

# ── Batch 2: Data science utilities — mirrors Kaggle cell 2 ──────────────
# !pip install -qU datasets pyarrow pyyaml tqdm pandas
_PACKAGES_DATA = [
    "datasets",
    "pyarrow",
    "pyyaml",
    "tqdm",
    "pandas",
    "sentencepiece",
    "accelerate",
]

for _batch in (_PACKAGES_ML, _PACKAGES_DATA):
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install",
            "--quiet",
            # Upgrade only when the installed version does NOT satisfy the
            # requirement — prevents pip from touching torch/torchvision.
            "--upgrade-strategy", "only-if-needed",
            "--upgrade",
            *_batch,
        ]
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Standard library imports (no ML yet — fast)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import shutil
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
IS_KAGGLE = True  # Set to False if testing locally

if IS_KAGGLE:
    INPUT_DIR = Path("/kaggle/input/datasets/pmtaiii/graphrag-final")
    OUTPUT_DIR = Path("/kaggle/working/sqlite_outputs")
    
    SQLITE_DIR = OUTPUT_DIR
    SQLITE_PATH = SQLITE_DIR / "phase6.sqlite"
    CHECKPOINT_PATH = Path("/kaggle/working/phase6_sqlite_checkpoint.json")
    
    # In Kaggle, we read the parquet files from the input dataset
    ENTITIES_PATH = INPUT_DIR / "entities.parquet"
    COMMUNITY_PATH = INPUT_DIR / "community_reports.parquet"
else:
    # Local fallback for testing
    HERE = Path(__file__).parent.parent
    OUTPUT = HERE / "output"
    SQLITE_DIR = OUTPUT / "sqlite_outputs"
    SQLITE_PATH = SQLITE_DIR / "phase6.sqlite"
    CHECKPOINT_PATH = OUTPUT / "phase6_sqlite_checkpoint.json"
    ENTITIES_PATH = OUTPUT / "entities.parquet"
    COMMUNITY_PATH = OUTPUT / "community_reports.parquet"
VECTOR_DIM = 768

TABLES: dict[str, dict[str, str]] = {
    "entity_description": {
        "schema": """
            CREATE TABLE IF NOT EXISTS entity_description (
                row_num INTEGER PRIMARY KEY,
                id TEXT,
                name TEXT,
                description TEXT,
                vector BLOB
            )
        """,
        "insert": "INSERT OR REPLACE INTO entity_description (row_num, id, name, description, vector) VALUES (?, ?, ?, ?, ?)",
    },
    "community_full_content": {
        "schema": """
            CREATE TABLE IF NOT EXISTS community_full_content (
                row_num INTEGER PRIMARY KEY,
                community INTEGER,
                title TEXT,
                full_content TEXT,
                vector BLOB
            )
        """,
        "insert": "INSERT OR REPLACE INTO community_full_content (row_num, community, title, full_content, vector) VALUES (?, ?, ?, ?, ?)",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase6_sqlite")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cp() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Checkpoint unreadable (%s) - starting fresh.", exc)
        return {}


def _save_cp(cp: dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2, ensure_ascii=False), encoding="utf-8")


def _cp_valid(cp: dict[str, Any]) -> bool:
    for key in TABLES:
        dim = cp.get(key, {}).get("vector_dim")
        if dim and dim != VECTOR_DIM:
            log.warning("Checkpoint vector_dim=%s for '%s' != %d - will reset.", dim, key, VECTOR_DIM)
            return False
    return True


def _reset() -> None:
    log.info("Resetting checkpoint and SQLite output...")
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        log.info("Deleted: %s", CHECKPOINT_PATH)
    if SQLITE_DIR.exists():
        shutil.rmtree(SQLITE_DIR)
        log.info("Deleted: %s", SQLITE_DIR)
    SQLITE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: Model loading (heavy imports deferred until after bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> Any:
    """Load DEk21_hcmute_embedding using SentenceTransformers.
    
    Includes the transformers AutoModel workaround exactly as used in
    parent_child.py to avoid dependency hell.
    """
    import torch
    
    from transformers import AutoModel
    if not getattr(AutoModel, '_dtype_patch_applied', False):
        _orig_from_pretrained = AutoModel.from_pretrained.__func__

        @classmethod
        def _patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            kwargs.pop('dtype', None)
            kwargs.pop('torch_dtype', None)
            return _orig_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)

        AutoModel.from_pretrained = _patched_from_pretrained
        AutoModel._dtype_patch_applied = True

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading DEk21 embedding model on %s...", device)

    model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding", device=device)
    log.info("DEk21 loaded. Output dim: %d", VECTOR_DIM)
    return model


def _segment(text: str) -> str:
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(text)


def _embed(model: Any, texts: list[str], max_length: int) -> list[list[float]]:
    if not texts:
        return []
    segmented = [_segment(t) for t in texts]
    embeddings = model.encode(
        segmented,
        batch_size=len(texts),
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


# ─────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ─────────────────────────────────────────────────────────────────────────────

class SqliteWriter:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._configure()
        self._create_schema()

    def _configure(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")

    def _create_schema(self) -> None:
        for spec in TABLES.values():
            self.conn.execute(spec["schema"])
        self.conn.commit()

    def table_names(self) -> list[str]:
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [row[0] for row in rows]

    def count_rows(self, table_name: str) -> int:
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        return int(cursor.fetchone()[0])

    def insert_rows(self, table_name: str, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        self.conn.executemany(TABLES[table_name]["insert"], rows)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding loops
# ─────────────────────────────────────────────────────────────────────────────

def _run_table(
    *,
    model: Any,
    writer: SqliteWriter,
    table_key: str,
    df: pd.DataFrame,
    text_col: str,
    fallback_col: str,
    row_builder,
    batch_size: int,
    max_length: int,
    cp: dict[str, Any],
) -> None:
    if cp.get(table_key, {}).get("status") == "completed":
        log.info("  '%s' already completed - skipping.", table_key)
        return

    total = len(df)
    done = writer.count_rows(table_key)
    log.info("  Rows to process: %d", total)

    if done >= total:
        log.info("  All rows already written - marking complete.")
        cp[table_key] = {
            "status": "completed",
            "completed_rows": total,
            "total_rows": total,
            "vector_dim": VECTOR_DIM,
        }
        _save_cp(cp)
        return

    remaining = df.iloc[done:].reset_index(drop=True)
    log.info("  Starting from row %d/%d", done, total)

    t0 = time.time()
    log_every = max(1, 500 // batch_size)

    for batch_index in range(0, len(remaining), batch_size):
        batch = remaining.iloc[batch_index : batch_index + batch_size]
        records = batch.to_dict(orient="records")

        texts = []
        for record in records:
            primary = str(record.get(text_col) or "").strip()
            fallback = str(record.get(fallback_col) or "").strip()
            texts.append(primary or fallback or "Khong co mo ta")

        vectors = _embed(model, texts, max_length=max_length)
        rows = row_builder(records, vectors, done + batch_index)
        writer.insert_rows(table_key, rows)

        done += len(records)
        cp[table_key] = {
            "status": "in_progress",
            "completed_rows": done,
            "total_rows": total,
            "vector_dim": VECTOR_DIM,
        }
        _save_cp(cp)

        current_batch = batch_index // batch_size
        if current_batch % log_every == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            log.info(
                "  [%s] %d/%d (%.1f%%) | %.1f rows/s | ETA %.0fs",
                table_key,
                done,
                total,
                100 * done / total,
                rate,
                eta,
            )

    cp[table_key] = {
        "status": "completed",
        "completed_rows": total,
        "total_rows": total,
        "vector_dim": VECTOR_DIM,
    }
    _save_cp(cp)
    log.info("  '%s' complete: %d rows.", table_key, total)


def embed_entities(model: Any, writer: SqliteWriter, batch_size: int, max_length: int, cp: dict[str, Any]) -> None:
    log.info("=" * 60)
    log.info("Phase 6A - entity_description")

    df = pd.read_parquet(ENTITIES_PATH, columns=["id", "name", "description"])
    df["id"] = df["id"].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    def _build_rows(records: list[dict[str, Any]], vectors: list[list[float]], start_row: int) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for offset, (record, vector) in enumerate(zip(records, vectors)):
            rows.append(
                (
                    start_row + offset,
                    record["id"],
                    record["name"],
                    record["description"],
                    _pack_vector(vector),
                )
            )
        return rows

    _run_table(
        model=model,
        writer=writer,
        table_key="entity_description",
        df=df,
        text_col="description",
        fallback_col="name",
        row_builder=_build_rows,
        batch_size=batch_size,
        max_length=max_length,
        cp=cp,
    )


def embed_communities(model: Any, writer: SqliteWriter, batch_size: int, max_length: int, cp: dict[str, Any]) -> None:
    log.info("=" * 60)
    log.info("Phase 6B - community_full_content")

    df = pd.read_parquet(COMMUNITY_PATH, columns=["community", "title", "full_content"])
    df["community"] = pd.to_numeric(df["community"], errors="coerce").fillna(0).astype("int64")
    df["title"] = df["title"].fillna("").astype(str)
    df["full_content"] = df["full_content"].fillna("").astype(str)

    def _build_rows(records: list[dict[str, Any]], vectors: list[list[float]], start_row: int) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        for offset, (record, vector) in enumerate(zip(records, vectors)):
            rows.append(
                (
                    start_row + offset,
                    int(record["community"]),
                    record["title"],
                    record["full_content"],
                    _pack_vector(vector),
                )
            )
        return rows

    _run_table(
        model=model,
        writer=writer,
        table_key="community_full_content",
        df=df,
        text_col="full_content",
        fallback_col="title",
        row_builder=_build_rows,
        batch_size=batch_size,
        max_length=max_length,
        cp=cp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify(writer: SqliteWriter) -> None:
    log.info("=" * 60)
    log.info("Verification:")
    all_ok = True
    for name in ("entity_description", "community_full_content"):
        if name in writer.table_names():
            n = writer.count_rows(name)
            log.info("  %-30s : %d rows", name, n)
        else:
            log.error("  %-30s : TABLE NOT FOUND", name)
            all_ok = False
    if not all_ok:
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: BGE-M3 embeddings -> SQLite")
    parser.add_argument("--reset", action="store_true", help="Delete checkpoint and SQLite output, start from scratch")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32)")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum token length passed to BGE-M3")

    # ── Notebook/Kaggle compatibility ────────────────────────────────────────
    # When executed inside a Jupyter kernel, sys.argv contains kernel-specific
    # flags like "-f /kernel_xyz.json" that argparse does not recognise and
    # will crash with SystemExit: 2.  Detect IPython and pass args=[] so
    # argparse uses only the defaults defined above.
    try:
        get_ipython  # type: ignore[name-defined]  # noqa: F821
        _in_notebook = True
    except NameError:
        _in_notebook = False

    args = parser.parse_args(args=[] if _in_notebook else None)

    log.info("=" * 60)
    log.info("Phase 6: DEk21 Embeddings -> SQLite")
    log.info("  SQLite : %s", SQLITE_PATH)
    log.info("  Dim    : %d", VECTOR_DIM)
    log.info("  Batch  : %d", args.batch_size)
    log.info("  MaxLen : %d", args.max_length)
    log.info("=" * 60)

    for path in (ENTITIES_PATH, COMMUNITY_PATH):
        if not path.exists():
            log.error("Required file not found: %s", path)
            sys.exit(1)

    cp = _load_cp()
    if args.reset or not _cp_valid(cp):
        _reset()
        cp = {}
    else:
        log.info("Valid checkpoint found - resuming.")

    model = load_model()
    writer = SqliteWriter(SQLITE_PATH)

    embed_entities(model, writer, args.batch_size, args.max_length, cp)
    embed_communities(model, writer, args.batch_size, args.max_length, cp)
    verify(writer)

    writer.close()

    log.info("=" * 60)
    log.info("Phase 6 complete! SQLite ready at: %s", SQLITE_PATH)


if __name__ == "__main__":
    main()