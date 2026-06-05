"""
local_reranker.py — Local Vietnamese Reranker service.

Model: thanhtantran/Vietnamese_Reranker (cross-encoder, BGE-M3 derivative)

Architecture & Scoring Notes
-----------------------------
This model outputs RAW LOGITS, not probabilities.  The logits for Vietnamese
legal text typically range from -6 to +2.  Applying sigmoid maps these to
the (0, 1) range, BUT the distribution is heavily left-skewed:

    Logit   →  Sigmoid
    ──────────────────
     -6.0   →  0.0025
     -4.0   →  0.0180
     -2.0   →  0.1192
     -1.0   →  0.2689
      0.0   →  0.5000
     +1.0   →  0.7311
     +2.0   →  0.8808

For typical legal queries, MOST docs score between -5 and -1 (sigmoid 0.01–0.27).
Only a very strong match scores above 0 (sigmoid > 0.5).

DESIGN DECISION: The reranker's job is to RANK (sort by relevance),
NOT to FILTER (drop low-scoring docs).  Quality gating is handled
DOWNSTREAM by `_should_fallback()` in routes.py.  This separates
concerns cleanly:
  - Reranker  → ranking / ordering
  - Fallback  → quality decision (Qdrant vs Tavily)
"""

import asyncio
import logging
import torch
import functools
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger("pipeline")


class LocalReranker:
    """
    Local Vietnamese Reranker service.
    Model: thanhtantran/Vietnamese_Reranker

    Usage:
        reranker = LocalReranker()
        reranker.initialize()
        ranked = await reranker.rerank(query, docs, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "thanhtantran/Vietnamese_Reranker",
        max_length: int = 2304,
        request_max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        # Runtime tokenization length — much shorter than model capability.
        # Legal chunks are 300-800 chars; 512 tokens is sufficient.
        # Reduces tokenization RAM by ~4.5x vs full 2304.
        self._request_max_length = request_max_length
        self.model = None
        self.tokenizer = None

    def initialize(self):
        import os
        num_threads = int(os.getenv("TORCH_NUM_THREADS", "2"))
        torch.set_num_threads(num_threads)
        log.info(f"Capping PyTorch CPU threads to {num_threads}")

        log.info(f"Loading local reranker model: {self.model_name} (FP32)")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.eval()
        log.info("✅ Local reranker loaded successfully.")

    def _sync_rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """
        Score and rank documents by relevance to query.

        NO threshold filtering — all docs are kept and sorted by score.
        The caller (routes.py _should_fallback) handles quality gating.

        Returns up to top_k docs sorted by descending relevance score.
        Scores are raw sigmoid probabilities in (0, 1) range.
        """
        if not docs or not self.model or not self.tokenizer:
            return docs

        # Prefer child_text (focused 400-token child snippet) when available.
        # Falls back to "text" for backward compatibility with the existing flat-chunk schema.
        # Under the parent-child architecture: child_text = what the model scores,
        # "text" = full parent context returned to the LLM after ranking.
        texts = [d.get("child_text") or d.get("text", "") for d in docs]
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            log.warning("[reranker] All %d docs have empty text — returning empty", len(docs))
            return []

        valid_texts = [texts[i] for i in valid_indices]
        pairs = [[query, text] for text in valid_texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self._request_max_length,
                return_tensors="pt",
            )
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()

        # Sigmoid maps logits to (0, 1) for consistent score interpretation.
        scores = torch.sigmoid(logits).tolist()
        raw_logits = logits.tolist()

        # Log full score distribution for diagnostics
        if scores:
            s_min = min(scores)
            s_max = max(scores)
            s_mean = sum(scores) / len(scores)
            log.info(
                "[reranker] Scored %d docs: "
                "sigmoid min=%.4f max=%.4f mean=%.4f | "
                "logit min=%.2f max=%.2f",
                len(scores), s_min, s_max, s_mean,
                min(raw_logits), max(raw_logits),
            )

        # Build ranked list — NO threshold filter, just rank by score
        ranked = []
        for i, score in enumerate(scores):
            original_idx = valid_indices[i]
            ranked.append({
                **docs[original_idx],
                "score": score,
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    async def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Async wrapper — runs sync inference in a thread executor.

        NOTE: No threshold parameter. The reranker ranks, not filters.
        Quality gating is handled by _should_fallback() in routes.py.
        """
        loop = asyncio.get_event_loop()
        func = functools.partial(self._sync_rerank, query, docs, top_k)
        return await loop.run_in_executor(None, func)
