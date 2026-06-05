#!/usr/bin/env python3
"""
enrich_qdrant_metadata.py — Join metadata.parquet fields into Qdrant point payloads.

Algorithm (3 phases):
  Phase 1 — Inspect : Load metadata.parquet, print schema, connect Qdrant.
  Phase 2 — Index   : Scroll ALL Qdrant points, build {doc_id → [point_ids]} map in memory.
  Phase 3 — Update  : For each doc_id found in BOTH metadata and Qdrant, call set_payload.

Join key : metadata.parquet["id"]  ==  Qdrant payload["doc_id"]
Skips    : doc_ids in metadata that have no matching points in Qdrant (logged as summary count).

Robustness:
  - Dry-run by default — pass --write to apply changes.
  - Auto-retry with exponential back-off on connection drops (WinError 10054).
  - Backpressure every 100 docs: forces wait=True so Qdrant can flush its buffer.
  - Idempotent: safe to re-run; set_payload overwrites with same values.

Usage:
    python data_pipelines/vector_rag/enrich_qdrant_metadata.py               # dry-run
    python data_pipelines/vector_rag/enrich_qdrant_metadata.py --write       # apply
    python data_pipelines/vector_rag/enrich_qdrant_metadata.py --write --batch-size 1000

Requirements:
    pip install qdrant-client pandas pyarrow
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

# ── Paths & defaults ──────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
METADATA_PATH = SCRIPT_DIR / "sqlite_outputs" / "metadata.parquet"
DEFAULT_URL   = "http://localhost:6333"
DEFAULT_COL   = "vietnamese_laws_m3"
SCROLL_BATCH      = 200     # points per scroll page — keep small to avoid Qdrant server-side timeout
SCROLL_SLEEP_SEC  = 0.15   # pause between scroll pages to reduce disk I/O pressure
SCROLL_MAX_RETRY  = 5      # retry attempts on scroll timeout
UPDATE_BATCH      = 500    # max point IDs per set_payload call
LOG_EVERY         = 50_000  # log progress every N points scrolled

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("enrich")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Load metadata
# ─────────────────────────────────────────────────────────────────────────────

def _load_metadata(path: Path) -> dict[int, dict[str, Any]]:
    """
    Load metadata.parquet → {int(id): {col: clean_value}}.

    - Casts id to int (handles float IDs from nullable-int parquet columns).
    - Converts NaN → "" for string fields, datetime → ISO string.
    - Excludes the join-key column 'id' from the returned payload dict.
    """
    log.info("Loading metadata: %s", path)
    df = pd.read_parquet(str(path))

    log.info("Metadata schema:")
    for col, dtype in df.dtypes.items():
        non_null = df[col].notna().sum()
        log.info("  %-35s %s  (non-null: %d / %d)", col, dtype, non_null, len(df))

    if "id" not in df.columns:
        log.error("'id' column not found in metadata.parquet. Available: %s", list(df.columns))
        sys.exit(1)

    payload_cols = [c for c in df.columns if c != "id"]
    result: dict[int, dict[str, Any]] = {}

    for row in df.itertuples(index=False):
        raw_id = getattr(row, "id")
        if raw_id is None or (isinstance(raw_id, float) and math.isnan(raw_id)):
            continue
        doc_id = int(raw_id)

        payload: dict[str, Any] = {}
        for col in payload_cols:
            val = getattr(row, col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                payload[col] = ""
            elif hasattr(val, "isoformat"):        # datetime / date / Timestamp
                payload[col] = val.isoformat()
            else:
                s = str(val).strip()
                payload[col] = "" if s == "nan" else s
        result[doc_id] = payload

    log.info("Metadata loaded: %d unique doc IDs", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Build Qdrant index
# ─────────────────────────────────────────────────────────────────────────────

def _build_qdrant_index(
    client: QdrantClient,
    collection: str,
) -> dict[int, list[int]]:
    """
    Scroll ALL points in the collection and build {doc_id → [point_id, ...]}.

    Uses small batch size + inter-batch sleep + per-page retry to avoid
    the Qdrant server-side 60s timeout that triggers on large on-disk collections.
    """
    log.info("Phase 2: Scrolling Qdrant '%s' to build doc_id index...", collection)
    total_pts = client.get_collection(collection).points_count
    log.info(
        "  Collection has %s points (batch=%d, sleep=%.2fs between pages).",
        f"{total_pts:,}", SCROLL_BATCH, SCROLL_SLEEP_SEC,
    )

    index: dict[int, list[int]] = defaultdict(list)
    scrolled = 0
    offset   = None
    t0       = time.monotonic()

    while True:
        # ── Retry loop per scroll page ─────────────────────────────────────
        batch = next_offset = None
        for attempt in range(1, SCROLL_MAX_RETRY + 1):
            try:
                batch, next_offset = client.scroll(
                    collection_name=collection,
                    limit=SCROLL_BATCH,
                    offset=offset,
                    with_payload=["doc_id"],
                    with_vectors=False,
                )
                break  # success
            except Exception as exc:
                if attempt == SCROLL_MAX_RETRY:
                    log.error(
                        "Scroll failed after %d attempts at offset=%s: %s",
                        SCROLL_MAX_RETRY, offset, exc,
                    )
                    raise
                delay = 2 ** attempt
                log.warning(
                    "  ⟳ Scroll error (%s). Retry %d/%d in %ds...",
                    type(exc).__name__, attempt, SCROLL_MAX_RETRY, delay,
                )
                time.sleep(delay)

        if not batch:
            break

        for pt in batch:
            raw = (pt.payload or {}).get("doc_id")
            if raw is not None:
                try:
                    index[int(raw)].append(pt.id)
                except (ValueError, TypeError):
                    pass

        scrolled += len(batch)
        if scrolled % LOG_EVERY < SCROLL_BATCH:
            elapsed = time.monotonic() - t0
            rate    = scrolled / elapsed if elapsed else 0
            log.info(
                "  Scrolled %s / ~%s points (%.0f pts/s)",
                f"{scrolled:,}", f"{total_pts:,}", rate,
            )

        if next_offset is None:
            break
        offset = next_offset

        # Brief pause to reduce disk I/O pressure on Qdrant
        time.sleep(SCROLL_SLEEP_SEC)

    elapsed = time.monotonic() - t0
    log.info(
        "  Index built: %s unique doc_ids from %s points in %.1fs",
        f"{len(index):,}", f"{scrolled:,}", elapsed,
    )
    return dict(index)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Update payloads
# ─────────────────────────────────────────────────────────────────────────────

def _update_payloads(
    client: QdrantClient,
    collection: str,
    metadata: dict[int, dict[str, Any]],
    qdrant_index: dict[int, list[int]],
    batch_size: int,
    dry_run: bool,
) -> None:
    """
    For each doc_id in BOTH metadata and qdrant_index, call set_payload.
    Skips doc_ids that only exist in metadata (logged as summary count).

    Robustness features:
      - Retry up to 5 times with exponential back-off on any exception.
      - Backpressure: every 100 doc updates, force wait=True to let Qdrant flush.
    """
    matched_ids   = sorted(set(metadata.keys()) & set(qdrant_index.keys()))
    skipped_count = len(metadata) - len(matched_ids)
    total_points  = sum(len(qdrant_index[d]) for d in matched_ids)

    log.info("Phase 3: Updating payloads")
    log.info("  Metadata doc_ids : %s", f"{len(metadata):,}")
    log.info("  Matched in Qdrant: %s doc_ids → %s points",
             f"{len(matched_ids):,}", f"{total_points:,}")
    log.info("  Not in Qdrant    : %s doc_ids (skipped)", f"{skipped_count:,}")

    if dry_run:
        log.info("  DRY-RUN — no changes written. Pass --write to apply.")
        for doc_id in matched_ids[:3]:
            log.info(
                "  [sample] doc_id=%d → point_ids=%s payload_keys=%s",
                doc_id, qdrant_index[doc_id][:3], list(metadata[doc_id].keys()),
            )
        return

    updated_docs   = 0
    updated_points = 0
    t0             = time.monotonic()
    MAX_RETRIES    = 5

    for doc_id in matched_ids:
        point_ids = qdrant_index[doc_id]
        payload   = metadata[doc_id]

        for i in range(0, len(point_ids), batch_size):
            chunk = point_ids[i : i + batch_size]

            # Backpressure: every 100 docs, switch to synchronous mode so
            # Qdrant has time to flush its write buffer before we send more.
            should_wait = (updated_docs > 0 and updated_docs % 100 == 0)

            # Retry loop — guards against WinError 10054 (connection forcibly closed)
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    client.set_payload(
                        collection_name=collection,
                        payload=payload,
                        points=chunk,
                        wait=should_wait,
                    )
                    break  # success
                except Exception as exc:
                    if attempt == MAX_RETRIES:
                        log.error(
                            "Failed to update doc_id=%d after %d attempts. Last error: %s",
                            doc_id, MAX_RETRIES, exc,
                        )
                        raise
                    delay = 2 ** attempt   # 2s, 4s, 8s, 16s
                    log.warning(
                        "  ⟳ Qdrant error (%s). Retrying %d/%d in %ds...",
                        type(exc).__name__, attempt, MAX_RETRIES, delay,
                    )
                    time.sleep(delay)

            updated_points += len(chunk)

        updated_docs += 1
        if updated_docs % 500 == 0:
            elapsed = time.monotonic() - t0
            log.info(
                "  Updated %s / %s doc_ids (%s points) in %.1fs",
                f"{updated_docs:,}", f"{len(matched_ids):,}",
                f"{updated_points:,}", elapsed,
            )

    elapsed = time.monotonic() - t0
    log.info(
        "✅ Done: %s doc_ids, %s points updated in %.1fs",
        f"{updated_docs:,}", f"{updated_points:,}", elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join metadata.parquet fields into Qdrant point payloads via doc_id.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata",   default=str(METADATA_PATH), help="Path to metadata.parquet")
    parser.add_argument("--qdrant-url", default=DEFAULT_URL,        help="Qdrant URL")
    parser.add_argument("--collection", default=DEFAULT_COL,        help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=UPDATE_BATCH,
                        help="Max point IDs per set_payload call")
    parser.add_argument("--write",      action="store_true",
                        help="Apply changes to Qdrant (default: dry-run, nothing is written)")
    args = parser.parse_args()

    dry_run = not args.write
    if dry_run:
        log.info("=" * 60)
        log.info("DRY-RUN MODE — pass --write to apply changes to Qdrant")
        log.info("=" * 60)

    # ── Phase 1: Load metadata ─────────────────────────────────────────────
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        log.error("metadata.parquet not found: %s", metadata_path)
        sys.exit(1)
    metadata = _load_metadata(metadata_path)

    # ── Connect Qdrant ─────────────────────────────────────────────────────
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    try:
        info = client.get_collection(args.collection)
        log.info(
            "✅ Connected to Qdrant: %s — collection '%s' (%s points)",
            args.qdrant_url, args.collection, f"{info.points_count:,}",
        )
    except Exception as exc:
        log.error("Cannot connect to Qdrant at %s: %s", args.qdrant_url, exc)
        sys.exit(1)

    # ── Phase 2: Build doc_id → [point_ids] index from Qdrant ─────────────
    qdrant_index = _build_qdrant_index(client, args.collection)

    # ── Phase 3: Update payloads ───────────────────────────────────────────
    _update_payloads(
        client=client,
        collection=args.collection,
        metadata=metadata,
        qdrant_index=qdrant_index,
        batch_size=args.batch_size,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
