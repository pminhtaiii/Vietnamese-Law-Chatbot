"""
migrate.py — Fixed Version
===========================
Root cause: Local Qdrant payload dùng key "content" (không phải "text").
            Script cũ dùng safe_payload.get("text") --> luôn trả về None.

Fixes:
  [F1] Đọc đúng field "content" từ local Qdrant payload
  [F2] Dùng đường dẫn tuyệt đối (absolute path) qua __file__ để tránh lỗi relative path
  [F3] Lưu lên Docker Qdrant với key "text" để khớp với retriever.py
"""

import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# [F2] Dùng __file__ để tạo path tuyệt đối bất kể CWD khi chạy script
SCRIPT_DIR    = Path(__file__).resolve().parent
LOCAL_DB_PATH = str(SCRIPT_DIR / "data")

DOCKER_URL      = "http://localhost:6333"
COLLECTION_NAME = "vietnamese_laws_m3"
BATCH_SIZE      = 500


def migrate_qdrant_data():
    print(f"[INFO] Script dir     : {SCRIPT_DIR}")
    print(f"[INFO] LOCAL_DB_PATH  : {LOCAL_DB_PATH}")
    print(f"[INFO] DOCKER_URL     : {DOCKER_URL}")
    print(f"[INFO] COLLECTION     : {COLLECTION_NAME}")

    # ── Kết nối local Qdrant ─────────────────────────────────────────────────
    if not Path(LOCAL_DB_PATH).exists():
        print(f"[ERROR] Không tìm thấy local DB tại: {LOCAL_DB_PATH}")
        return

    print(f"\n[INFO] Đang kết nối local Qdrant...")
    local_client = QdrantClient(path=LOCAL_DB_PATH)

    try:
        collection_info = local_client.get_collection(COLLECTION_NAME)
        print(f"[INFO] Local collection có {collection_info.points_count:,} points")
    except Exception as e:
        print(f"[ERROR] Không đọc được local collection: {e}")
        return

    # ── Kết nối Docker Qdrant ─────────────────────────────────────────────────
    try:
        docker_client = QdrantClient(url=DOCKER_URL, timeout=60.0)
        docker_client.get_collections()
        print(f"[INFO] Đã kết nối Docker Qdrant tại {DOCKER_URL}")
    except Exception as e:
        print(f"[ERROR] Lỗi kết nối Docker Qdrant: {e}")
        return

    # ── Tạo collection trên Docker nếu chưa có ───────────────────────────────
    vectors_config        = collection_info.config.params.vectors
    sparse_vectors_config = collection_info.config.params.sparse_vectors

    if not docker_client.collection_exists(COLLECTION_NAME):
        docker_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        print(f"[INFO] Đã tạo collection '{COLLECTION_NAME}' trên Docker Qdrant.")
    else:
        print(f"[INFO] Collection '{COLLECTION_NAME}' đã tồn tại trên Docker — sẽ upsert.")

    # ── Scroll → migrate ──────────────────────────────────────────────────────
    offset         = None
    total_migrated = 0
    total_no_text  = 0

    print(f"\n[INFO] Bắt đầu migrate (batch_size={BATCH_SIZE})...")

    while True:
        records, offset = local_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )

        if not records:
            break

        points = []
        for r in records:
            safe_payload = r.payload or {}

            # [F1] KEY FIX: local Qdrant lưu text dưới key "content", không phải "text"
            text = safe_payload.get("content") or safe_payload.get("text") or ""
            cid  = safe_payload.get("cid")

            if not text:
                total_no_text += 1

            # [F3] Lưu lên Docker với key "text" để retriever.py đọc được
            clean_payload = {
                "cid":  cid,
                "text": text,
            }

            points.append(
                models.PointStruct(
                    id=r.id,
                    vector=r.vector,
                    payload=clean_payload,
                )
            )

        docker_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        total_migrated += len(points)
        print(f"  → Migrated: {total_migrated:,} points | Empty text: {total_no_text}")

        if offset is None:
            break

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"✅ HOÀN TẤT MIGRATE")
    print(f"   Tổng points migrated : {total_migrated:,}")
    print(f"   Points không có text : {total_no_text:,}")
    print("=" * 60)

    if total_no_text > 0:
        print(f"\n[WARN] {total_no_text} points thiếu text — kiểm tra lại local DB.")


if __name__ == "__main__":
    migrate_qdrant_data()