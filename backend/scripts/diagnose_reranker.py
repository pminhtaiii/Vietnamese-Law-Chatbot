"""
diagnose_reranker.py — Score distribution diagnostic for thanhtantran/Vietnamese_Reranker

Sends a set of test queries to the live Qdrant instance, retrieves raw candidates,
then scores them with the local reranker and prints the full score distribution.

This reveals whether the reranker is the bottleneck (all scores < 0.3 → everything
gets filtered out → empty docs → Tavily fallback).

Usage:
    docker exec -it backend-api python scripts/diagnose_reranker.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from FlagEmbedding import BGEM3FlagModel

from app.core.config import settings


TEST_QUERIES = [
    "mức phạt nồng độ cồn",
    "thủ tục đăng ký kết hôn",
    "điều kiện thành lập công ty TNHH",
    "nghỉ thai sản bao nhiêu tháng",
    "quy định về phòng cháy chữa cháy",
]


async def get_raw_candidates(query: str, embedder, qdrant, top_k: int = 10):
    """Mimic retriever._hybrid_search for a single query."""
    output = embedder.encode(
        [query], batch_size=1,
        return_dense=True, return_sparse=True,
        return_colbert_vecs=False, max_length=512,
    )
    dense_vec = output["dense_vecs"][0].tolist()
    sparse_dict = output["lexical_weights"][0]
    sparse_indices = [int(k) for k in sparse_dict.keys()]
    sparse_values = list(sparse_dict.values())

    result = await qdrant.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            qmodels.Prefetch(query=dense_vec, using="dense", limit=top_k * 2),
            qmodels.Prefetch(
                query=qmodels.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse", limit=top_k * 2,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    docs = []
    for pt in result.points:
        payload = pt.payload or {}
        docs.append({
            "cid": str(payload.get("cid", pt.id)),
            "text": payload.get("text", ""),
            "qdrant_rrf_score": pt.score,
        })
    return docs


def rerank_with_scores(query: str, docs: list, tokenizer, model, max_length: int = 512):
    """Score each doc and return raw logits + sigmoid scores."""
    texts = [d.get("child_text") or d.get("text", "") for d in docs]
    pairs = [[query, t] for t in texts if t.strip()]

    if not pairs:
        return []

    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        logits = model(**inputs, return_dict=True).logits.view(-1).float()
        sigmoid_scores = torch.sigmoid(logits).tolist()
        raw_logits = logits.tolist()

    results = []
    for i, (logit, sig) in enumerate(zip(raw_logits, sigmoid_scores)):
        results.append({
            "cid": docs[i]["cid"],
            "text_preview": docs[i]["text"][:120].replace("\n", " "),
            "qdrant_rrf": docs[i]["qdrant_rrf_score"],
            "raw_logit": round(logit, 4),
            "sigmoid_score": round(sig, 4),
            "passes_0.3": sig >= 0.3,
            "passes_0.1": sig >= 0.1,
            "passes_0.05": sig >= 0.05,
        })
    return results


async def main():
    print("=" * 80)
    print("RERANKER SCORE DISTRIBUTION DIAGNOSTIC")
    print("=" * 80)

    # Load models
    print("\n📦 Loading BGE-M3 embedder...")
    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    print("📦 Loading Vietnamese_Reranker...")
    tokenizer = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(settings.RERANKER_MODEL)
    model.eval()

    print("📦 Connecting to Qdrant...")
    qdrant = AsyncQdrantClient(url=settings.QDRANT_HOST, timeout=30.0)
    info = await qdrant.get_collection(settings.COLLECTION_NAME)
    print(f"   Collection: {settings.COLLECTION_NAME} ({info.points_count:,} points)")

    print("\n" + "=" * 80)

    for query in TEST_QUERIES:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 70)

        # Step 1: Get raw candidates from Qdrant
        docs = await get_raw_candidates(query, embedder, qdrant, top_k=10)
        print(f"   Qdrant returned: {len(docs)} docs")

        if not docs:
            print("   ❌ NO DOCS FROM QDRANT — problem is upstream, not reranker")
            continue

        # Step 2: Score with reranker
        scored = rerank_with_scores(query, docs, tokenizer, model)

        # Step 3: Print full distribution
        print(f"\n   {'CID':>8} | {'Logit':>8} | {'Sigmoid':>8} | {'≥0.3':>5} | {'≥0.1':>5} | {'≥0.05':>5} | Text Preview")
        print(f"   {'─'*8} | {'─'*8} | {'─'*8} | {'─'*5} | {'─'*5} | {'─'*5} | {'─'*40}")

        pass_03 = 0
        pass_01 = 0
        pass_005 = 0

        for r in scored:
            flag_03 = "✅" if r["passes_0.3"] else "❌"
            flag_01 = "✅" if r["passes_0.1"] else "❌"
            flag_005 = "✅" if r["passes_0.05"] else "❌"
            if r["passes_0.3"]: pass_03 += 1
            if r["passes_0.1"]: pass_01 += 1
            if r["passes_0.05"]: pass_005 += 1

            print(f"   {r['cid']:>8} | {r['raw_logit']:>8.4f} | {r['sigmoid_score']:>8.4f} | {flag_03:>5} | {flag_01:>5} | {flag_005:>5} | {r['text_preview'][:50]}")

        sigs = [r["sigmoid_score"] for r in scored]
        mean_sig = sum(sigs) / len(sigs) if sigs else 0
        print(f"\n   📊 Score stats: min={min(sigs):.4f}  max={max(sigs):.4f}  mean={mean_sig:.4f}")
        print(f"   📊 Pass rates:  ≥0.3: {pass_03}/{len(scored)}  |  ≥0.1: {pass_01}/{len(scored)}  |  ≥0.05: {pass_005}/{len(scored)}")

        if pass_03 == 0:
            print(f"   ⚠️  ALL docs filtered at threshold=0.3 → reranker returns EMPTY → Tavily fallback!")
        if mean_sig < 0.05:
            print(f"   ⚠️  Mean score {mean_sig:.4f} < 0.05 → even _should_fallback triggers!")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

    await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
