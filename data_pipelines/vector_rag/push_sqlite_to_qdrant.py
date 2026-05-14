"""
push_sqlite_to_qdrant.py — SQLite → Docker Qdrant uploader
=============================================================
Đọc các file chunks_XXXX.sqlite (output của dataset_replace_kaggle.ipynb)
và đẩy lên Qdrant Docker container.

Luồng dữ liệu:
  SQLite BLOB (dense_vector)  → struct.unpack → list[float] → Qdrant NamedVector
  SQLite TEXT (sparse_indices) → json.loads   → list[int]
  SQLite TEXT (sparse_values)  → json.loads   → list[float]  → Qdrant SparseVector

Qdrant collection schema (khớp với retriever.py):
  Point ID   : int (= cid)
  Vectors    : {"dense": 1024-dim COSINE, "sparse": SparseVector}
  Payload    : {"text": str, "cid": int}

Tính năng:
  - Tự động tạo collection nếu chưa tồn tại.
  - Upsert theo batch (mặc định 200 points/batch).
  - Checkpoint: ghi lại file sqlite + offset đã xử lý xong để resume.
  - Chạy lại an toàn: bỏ qua file đã hoàn thành, tiếp tục từ chỗ dừng.

Cách dùng:
  python push_sqlite_to_qdrant.py                          # mặc định
  python push_sqlite_to_qdrant.py --sqlite-dir D:/my_data  # chỉ định thư mục
  python push_sqlite_to_qdrant.py --qdrant-url http://localhost:6333
  python push_sqlite_to_qdrant.py --reset                  # xoá collection cũ, chạy lại từ đầu
"""

import argparse
import json
import logging
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SQLITE_DIR    = str(SCRIPT_DIR / "sqlite_outputs")
DEFAULT_QDRANT_URL    = "http://localhost:6333"
DEFAULT_COLLECTION    = "vietnamese_laws_m3"
DEFAULT_BATCH_SIZE    = 500
DEFAULT_EMBEDDING_DIM = 1024
CHECKPOINT_FILENAME   = "push_checkpoint.json"

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("push_sqlite")


# ─── CHECKPOINT ───────────────────────────────────────────────────────────────

class PushCheckpoint:
    """Track which SQLite files (and how many rows) have been pushed."""

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path
        self.data: dict = {
            "completed_files": [],    # list of filenames fully pushed
            "partial_file": None,     # filename currently in progress
            "partial_offset": 0,      # how many rows already pushed from that file
            "total_points_pushed": 0,
        }

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        log.info(
            "Checkpoint loaded: %d files done, partial=%s offset=%d, total_pushed=%d",
            len(self.data.get("completed_files", [])),
            self.data.get("partial_file"),
            self.data.get("partial_offset", 0),
            self.data.get("total_points_pushed", 0),
        )
        return True

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def is_file_done(self, filename: str) -> bool:
        return filename in self.data.get("completed_files", [])

    def get_partial_offset(self, filename: str) -> int:
        if self.data.get("partial_file") == filename:
            return self.data.get("partial_offset", 0)
        return 0

    def mark_batch_done(self, filename: str, new_offset: int, batch_count: int):
        self.data["partial_file"] = filename
        self.data["partial_offset"] = new_offset
        self.data["total_points_pushed"] = (
            self.data.get("total_points_pushed", 0) + batch_count
        )
        self.save()

    def mark_file_done(self, filename: str):
        completed = self.data.get("completed_files", [])
        if filename not in completed:
            completed.append(filename)
        self.data["completed_files"] = completed
        self.data["partial_file"] = None
        self.data["partial_offset"] = 0
        self.save()

    def reset(self):
        self.data = {
            "completed_files": [],
            "partial_file": None,
            "partial_offset": 0,
            "total_points_pushed": 0,
        }
        if self.path.exists():
            self.path.unlink()


# ─── VECTOR UNPACKING ────────────────────────────────────────────────────────

def unpack_dense_blob(blob: bytes, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """
    Unpack dense vector from SQLite BLOB.

    The blob was created with: struct.pack(f'{len(dense)}f', *dense)
    So each float is 4 bytes (float32), total = dim * 4 bytes.
    """
    expected_size = dim * 4
    if len(blob) != expected_size:
        raise ValueError(
            f"Dense BLOB size mismatch: got {len(blob)} bytes, "
            f"expected {expected_size} (dim={dim})"
        )
    return list(struct.unpack(f"{dim}f", blob))


def parse_sparse_vector(
    indices_json: str, values_json: str
) -> Optional[models.SparseVector]:
    """Parse sparse vector from JSON-encoded indices and values."""
    try:
        indices = json.loads(indices_json)
        values = json.loads(values_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if not indices or not values or len(indices) != len(values):
        return None

    return models.SparseVector(
        indices=[int(i) for i in indices],
        values=[float(v) for v in values],
    )


# ─── COLLECTION SETUP ────────────────────────────────────────────────────────

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dim: int = DEFAULT_EMBEDDING_DIM,
    reset: bool = False,
):
    """Create the Qdrant collection if it doesn't exist (or reset it).

    Collection features:
      - Dense vectors (COSINE) with Scalar Quantization (INT8):
        Reduces RAM 4x per vector while keeping float32 on disk for rescoring.
        Recall impact: <1% with rescore=True at query time.
      - Sparse vectors for lexical matching (BM25-style).
      - HNSW indexing_threshold raised to 20000 for faster bulk upsert
        (defers index building until enough points accumulate).
    """
    if reset and client.collection_exists(collection_name):
        log.warning("--reset flag: deleting existing collection '%s'", collection_name)
        client.delete_collection(collection_name)

    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        log.info(
            "Collection '%s' already exists (%s points)",
            collection_name,
            f"{info.points_count:,}" if info.points_count else "0",
        )
        return

    log.info("Creating collection '%s' (dim=%d, COSINE + sparse, SQ int8)", collection_name, dim)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,
        ),
    )
    log.info("Collection '%s' created with Scalar Quantization ✓", collection_name)


# ─── MAIN PUSH LOGIC ─────────────────────────────────────────────────────────

def push_one_sqlite(
    db_path: Path,
    client: QdrantClient,
    collection_name: str,
    checkpoint: PushCheckpoint,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dim: int = DEFAULT_EMBEDDING_DIM,
) -> int:
    """
    Read one SQLite file and upsert all rows to Qdrant.
    Returns total points pushed from this file.
    """
    filename = db_path.name

    if checkpoint.is_file_done(filename):
        log.info("  ⏭  %s — already done (skipped)", filename)
        return 0

    start_offset = checkpoint.get_partial_offset(filename)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Count total rows
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_rows = cursor.fetchone()[0]

    if start_offset >= total_rows:
        checkpoint.mark_file_done(filename)
        conn.close()
        log.info("  ✓  %s — %d rows (all already pushed)", filename, total_rows)
        return 0

    log.info(
        "  📂 %s — %d rows total, resuming from offset %d",
        filename, total_rows, start_offset,
    )

    # Fetch rows with OFFSET for resume
    cursor.execute(
        "SELECT * FROM chunks LIMIT -1 OFFSET ?",
        (start_offset,),
    )

    pushed_this_file = 0
    current_offset = start_offset
    batch_points: list[models.PointStruct] = []
    errors = 0

    for row in cursor:
        try:
            cid = int(row["cid"])
            text = str(row["text"] or "")

            # Unpack dense vector
            dense_blob = row["dense_vector"]
            if not dense_blob:
                errors += 1
                continue
            dense_vec = unpack_dense_blob(dense_blob, dim)

            # Parse sparse vector
            sparse_vec = parse_sparse_vector(
                row["sparse_indices"], row["sparse_values"]
            )

            # Build vector dict
            vectors = {"dense": dense_vec}
            if sparse_vec:
                vectors["sparse"] = sparse_vec

            # Payload bao gồm cả những field gốc (nếu có) và field dịch (nếu có)
            # Khớp với schema của retriever.py và bổ sung thêm metadata phong phú
            payload = {
                "cid": cid,
                "text": text,
                # Mapping các cột tiếng Anh đang có sẵn trong SQLite hiện tại
                "title": str(row["title"] or "") if "title" in row.keys() else "",
                "so_ky_hieu": str(row["document_number"] or "") if "document_number" in row.keys() else "",
                "loai_van_ban": str(row["legal_type"] or "") if "legal_type" in row.keys() else "",
                "linh_vuc": str(row["legal_sectors"] or "") if "legal_sectors" in row.keys() else "",
                "co_quan_ban_hanh": str(row["issuing_authority"] or "") if "issuing_authority" in row.keys() else "",
                "ngay_ban_hanh": str(row["issuance_date"] or "") if "issuance_date" in row.keys() else "",
                "nguon_thu_thap": str(row["url"] or "") if "url" in row.keys() else "",
                
                # Mapping các cột tiếng Việt bổ sung (nếu bạn update Kaggle script để lưu thêm)
                "doc_id": int(row["doc_id"]) if "doc_id" in row.keys() else None,
                "ngay_co_hieu_luc": str(row["ngay_co_hieu_luc"] or "") if "ngay_co_hieu_luc" in row.keys() else "",
                "ngay_het_hieu_luc": str(row["ngay_het_hieu_luc"] or "") if "ngay_het_hieu_luc" in row.keys() else "",
                "ngay_dang_cong_bao": str(row["ngay_dang_cong_bao"] or "") if "ngay_dang_cong_bao" in row.keys() else "",
                "nganh": str(row["nganh"] or "") if "nganh" in row.keys() else "",
                "chuc_danh": str(row["chuc_danh"] or "") if "chuc_danh" in row.keys() else "",
                "nguoi_ky": str(row["nguoi_ky"] or "") if "nguoi_ky" in row.keys() else "",
                "pham_vi": str(row["pham_vi"] or "") if "pham_vi" in row.keys() else "",
                "thong_tin_ap_dung": str(row["thong_tin_ap_dung"] or "") if "thong_tin_ap_dung" in row.keys() else "",
                "tinh_trang_hieu_luc": str(row["tinh_trang_hieu_luc"] or "") if "tinh_trang_hieu_luc" in row.keys() else "",
            }

            batch_points.append(
                models.PointStruct(
                    id=cid,
                    vector=vectors,
                    payload=payload,
                )
            )

        except Exception as exc:
            errors += 1
            if errors <= 5:
                log.warning("    Error at offset %d: %s", current_offset, exc)
            continue

        # Flush batch
        if len(batch_points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=batch_points,
                wait=True,
            )
            pushed_this_file += len(batch_points)
            current_offset += len(batch_points)
            checkpoint.mark_batch_done(filename, current_offset, len(batch_points))
            batch_points.clear()

            if pushed_this_file % 2000 == 0:
                log.info(
                    "    → %s: %d/%d pushed",
                    filename, current_offset, total_rows,
                )

    # Flush remaining
    if batch_points:
        client.upsert(
            collection_name=collection_name,
            points=batch_points,
            wait=True,
        )
        pushed_this_file += len(batch_points)
        current_offset += len(batch_points)

    checkpoint.mark_file_done(filename)
    conn.close()

    log.info(
        "  ✓  %s — pushed %d points (%d errors)",
        filename, pushed_this_file, errors,
    )
    return pushed_this_file


def main():
    parser = argparse.ArgumentParser(
        description="Push SQLite embedding files to Qdrant Docker"
    )
    parser.add_argument(
        "--sqlite-dir", default=DEFAULT_SQLITE_DIR,
        help=f"Directory containing chunks_XXXX.sqlite files (default: {DEFAULT_SQLITE_DIR})",
    )
    parser.add_argument(
        "--qdrant-url", default=DEFAULT_QDRANT_URL,
        help=f"Qdrant server URL (default: {DEFAULT_QDRANT_URL})",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Points per upsert batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dim", type=int, default=DEFAULT_EMBEDDING_DIM,
        help=f"Dense vector dimension (default: {DEFAULT_EMBEDDING_DIM})",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing collection and checkpoint, start fresh",
    )
    args = parser.parse_args()

    sqlite_dir = Path(args.sqlite_dir)
    if not sqlite_dir.exists():
        log.error("SQLite directory not found: %s", sqlite_dir)
        sys.exit(1)

    # Find all sqlite files, sorted
    sqlite_files = sorted(sqlite_dir.glob("chunks_*.sqlite"))
    if not sqlite_files:
        log.error("No chunks_*.sqlite files found in %s", sqlite_dir)
        sys.exit(1)

    log.info("Found %d SQLite file(s) in %s", len(sqlite_files), sqlite_dir)

    # Connect to Qdrant
    try:
        client = QdrantClient(url=args.qdrant_url, timeout=120.0)
        client.get_collections()  # connectivity test
        log.info("Connected to Qdrant at %s ✓", args.qdrant_url)
    except Exception as exc:
        log.error("Cannot connect to Qdrant at %s: %s", args.qdrant_url, exc)
        sys.exit(1)

    # Checkpoint
    checkpoint_path = sqlite_dir / CHECKPOINT_FILENAME
    checkpoint = PushCheckpoint(checkpoint_path)

    if args.reset:
        checkpoint.reset()

    checkpoint.load()

    # Ensure collection
    ensure_collection(client, args.collection, dim=args.dim, reset=args.reset)

    # Push each file
    t_start = time.time()
    grand_total = 0

    for db_path in sqlite_files:
        pushed = push_one_sqlite(
            db_path=db_path,
            client=client,
            collection_name=args.collection,
            checkpoint=checkpoint,
            batch_size=args.batch_size,
            dim=args.dim,
        )
        grand_total += pushed

    elapsed = time.time() - t_start

    # Final summary
    info = client.get_collection(args.collection)
    print()
    print("=" * 60)
    print("✅ PUSH HOÀN TẤT")
    print(f"   SQLite files processed : {len(sqlite_files)}")
    print(f"   Points pushed (session): {grand_total:,}")
    print(f"   Total in collection    : {info.points_count:,}")
    print(f"   Time elapsed           : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"   Checkpoint             : {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
