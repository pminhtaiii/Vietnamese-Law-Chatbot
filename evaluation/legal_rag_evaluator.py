#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
legal_rag_evaluator.py
======================
End-to-end evaluation pipeline for the Vietnamese Legal RAG Chatbot.

Evaluates two core capabilities:
  1. **Retrieval Quality** — Does the retriever find the ground-truth context?
  2. **Generation Quality** — Does the generator produce faithful, relevant answers?

Uses LLM-as-a-Judge (Cohere Command A Reasoning) to score Faithfulness, Answer Relevancy,
and Context Utilization.  Generation is done via Gemini.  The two APIs use
separate keys to avoid conflicts with the backend reranker.
Ground-truth data comes from `train.csv`.

Usage (standalone — no running backend required)
-------------------------------------------------
    # Full run (default 1000 samples):
    python legal_rag_evaluator.py --gemini_key YOUR_KEY --cohere_key YOUR_KEY

    # Custom sample size + output dir:
    python legal_rag_evaluator.py \
        --gemini_key YOUR_KEY \
        --cohere_key YOUR_KEY \
        --n_samples 500 \
        --output_dir eval_results

    # Quick smoke test (10 samples, generation-only mode):
    python legal_rag_evaluator.py --gemini_key YOUR_KEY --cohere_key YOUR_KEY --n_samples 10 --no_backend

Dependencies
------------
    pip install google-generativeai cohere pandas tqdm httpx

Author: Evaluation Pipeline for Law Chatbot
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
import httpx
import pandas as pd
from tqdm.auto import tqdm
import cohere

try:
    from dotenv import load_dotenv
    load_dotenv(".env.eval")
except ImportError:
    pass

# MLflow is optional — script works fine without it.
try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration & Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# evaluation/ lives one level below the project root
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "raw_csv" / "train.csv"
DEFAULT_OUTPUT_DIR = EVAL_DIR / "eval_results"
DEFAULT_N_SAMPLES = 50
DEFAULT_JUDGE_MODEL = "command-a-03-2025"
DEFAULT_GENERATOR_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_RERANKER_MODEL = "rerank-multilingual-v3.0"
DEFAULT_TOP_K = 5

# Rate limiting for Gemini API (generator)
GEMINI_RPM_LIMIT = 14            # Conservative for free tier
GEMINI_DELAY_SEC = 60.0 / GEMINI_RPM_LIMIT
GEMINI_CONCURRENT_LIMIT = 2       # Max concurrent Gemini API calls

# Rate limiting for Cohere API (judge)
COHERE_RPM_LIMIT = 18              # Cohere trial tier: ~20 RPM
COHERE_DELAY_SEC = 60.0 / COHERE_RPM_LIMIT
COHERE_CONCURRENT_LIMIT = 2        # Max concurrent Cohere API calls

RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 2.0

# Prompt/score normalization
PROMPT_QUESTION_CHAR_LIMIT = 700
PROMPT_CONTEXT_CHAR_LIMIT = 2500
PROMPT_ANSWER_CHAR_LIMIT = 1800
JUDGE_SCORE_MIN = 1.0
JUDGE_SCORE_MAX = 5.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("LegalEvaluator")


class AsyncRateLimiter:
    """Simple global async rate limiter to enforce min spacing between API calls."""

    def __init__(self, min_interval_sec: float):
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._lock = asyncio.Lock()
        self._next_allowed_ts = 0.0

    async def wait_turn(self) -> None:
        if self.min_interval_sec <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            if now < self._next_allowed_ts:
                await asyncio.sleep(self._next_allowed_ts - now)
                now = time.monotonic()
            self._next_allowed_ts = max(self._next_allowed_ts, now) + self.min_interval_sec


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class GoldenSample:
    """A single Q&A evaluation sample from train.csv."""
    qid: int
    question: str
    ground_truth_context: str       # The correct legal text from train.csv
    ground_truth_cids: List[int]    # The correct document IDs


@dataclass
class RetrievalResult:
    """Output from the retrieval stage."""
    retrieved_texts: List[str]
    retrieved_cids: List[int]
    retrieved_scores: List[float]
    latency_ms: float
    source: str = "backend"         # backend | oracle_gt | backend_error


@dataclass
class GenerationResult:
    """Output from the generation stage."""
    answer: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class EvalScores:
    """LLM-as-a-Judge evaluation scores for a single sample."""
    # Retrieval Metrics
    retrieval_metrics_available: bool = False
    context_hit: bool = False           # Did retriever find the GT context?
    context_cid_recall: float = 0.0     # Fraction of GT CIDs found in retrieved

    # Generation Metrics (scored by LLM Judge, 1-5 scale)
    faithfulness: float = 0.0           # Answer stays within retrieved context
    answer_relevancy: float = 0.0       # Answer addresses the question
    completeness: float = 0.0           # Answer covers all key legal points
    citation_accuracy: float = 0.0      # Citations are correct & present

    # Composite
    overall_score: float = 0.0

    judge_reasoning: str = ""


@dataclass
class EvalRecord:
    """Complete evaluation record combining all stages."""
    sample: GoldenSample
    retrieval: Optional[RetrievalResult] = None
    generation: Optional[GenerationResult] = None
    scores: Optional[EvalScores] = None
    error: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Golden Dataset Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GoldenDatasetBuilder:
    """Extracts and samples evaluation data from train.csv."""

    def __init__(self, csv_path: Path, n_samples: int = DEFAULT_N_SAMPLES, seed: int = 42):
        self.csv_path = csv_path
        self.n_samples = n_samples
        self.seed = seed

    def _parse_context(self, raw: str) -> str:
        """Parse the list-string context format from train.csv."""
        if not isinstance(raw, str) or not raw.strip():
            return ""
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return "\n\n".join(str(item).strip() for item in parsed if str(item).strip())
        except (ValueError, SyntaxError):
            pass
        return raw.strip()

    def _parse_cids(self, raw: str) -> List[int]:
        """Parse CID field which can be '[123]' or '[123 456]'."""
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [int(c) for c in parsed]
            return [int(parsed)]
        except (ValueError, SyntaxError):
            # Try splitting by space
            nums = re.findall(r"\d+", str(raw))
            return [int(n) for n in nums]

    def build(self) -> List[GoldenSample]:
        """Load train.csv, clean, sample, and return golden samples."""
        log.info(f"📂 Loading golden dataset from: {self.csv_path}")
        df = pd.read_csv(self.csv_path, dtype=str)
        log.info(f"   Total rows in CSV: {len(df):,}")

        # Clean and filter
        df = df.dropna(subset=["question", "context"])
        df = df[df["question"].str.strip().str.len() > 10]
        df = df[df["context"].str.strip().str.len() > 20]
        log.info(f"   After cleaning: {len(df):,} valid rows")

        # Stratified-ish sampling: prefer diverse question types
        n = min(self.n_samples, len(df))
        sampled = df.sample(n=n, random_state=self.seed)
        log.info(f"   Sampled: {n:,} rows (seed={self.seed})")

        samples = []
        for _, row in sampled.iterrows():
            s = GoldenSample(
                qid=int(row.get("qid", 0) or 0),
                question=row["question"].strip(),
                ground_truth_context=self._parse_context(row["context"]),
                ground_truth_cids=self._parse_cids(str(row.get("cid", "[]"))),
            )
            samples.append(s)

        log.info(f"   ✅ Golden dataset ready: {len(samples):,} samples")
        return samples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Retrieval Stage (calls the actual backend retriever)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RetrieverClient:
    """
    Calls the running backend API for retrieval.
    Falls back to a 'pass-through' mode using GT context if backend is offline.
    """

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        top_k: int = DEFAULT_TOP_K,
        use_backend: bool = True,
    ):
        self.backend_url = backend_url.rstrip("/")
        self.top_k = top_k
        self.use_backend = use_backend
        self._backend_alive = False

    async def _check_backend(self) -> bool:
        """Ping backend to see if it's running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Prefer health endpoint, then fall back to docs if needed.
                for path in ("/health", "/docs"):
                    try:
                        resp = await client.get(f"{self.backend_url}{path}")
                        if resp.status_code == 200:
                            self._backend_alive = True
                            return True
                    except Exception:
                        continue
                self._backend_alive = False
        except Exception:
            self._backend_alive = False
        return self._backend_alive

    async def retrieve(self, question: str, gt_context: str = "", gt_cids: List[int] = None) -> RetrievalResult:
        """
        Retrieve documents for a question.
        If backend is alive → real retrieval.
        Otherwise → use ground-truth context (for generation-only evaluation).
        """
        if self.use_backend and self._backend_alive:
            return await self._retrieve_from_backend(question)
        else:
            # Pass-through mode: use GT context for generation eval
            return RetrievalResult(
                retrieved_texts=[gt_context] if gt_context else [],
                retrieved_cids=gt_cids or [],
                retrieved_scores=[1.0] if gt_context else [],
                latency_ms=0.0,
                source="oracle_gt",
            )

    async def _retrieve_from_backend(self, question: str) -> RetrievalResult:
        """Call the /api/retrieve endpoint (retrieval-only, no generation)."""
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    f"{self.backend_url}/api/retrieve",
                    json={
                        "message":         question,
                        "top_k":           self.top_k,
                        "include_content": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                # Prefer the backend-reported retrieval_ms; fall back to
                # our own wall-clock measurement so the field is never None.
                elapsed = data.get("retrieval_ms") or (time.perf_counter() - start) * 1000

                sources = data.get("sources", [])
                retrieved_texts = [s.get("content", "") for s in sources]
                retrieved_cids = []
                for s in sources:
                    doc_id = str(s.get("id", "-1"))
                    retrieved_cids.append(int(doc_id) if doc_id.lstrip("-").isdigit() else -1)

                return RetrievalResult(
                    retrieved_texts=retrieved_texts,
                    retrieved_cids=retrieved_cids,
                    retrieved_scores=[s.get("score", 0.0) for s in sources],
                    latency_ms=elapsed,
                    source="backend",
                )
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                log.warning(f"Backend retrieval failed: {e}")
                raise RuntimeError(
                    f"Backend retrieval failed after {elapsed:.0f}ms: {e}"
                ) from e


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Generation Stage (direct Gemini call, mirrors backend generator)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_INSTRUCTION = """Bạn là một chuyên gia tư vấn Pháp luật Việt Nam xuất sắc, tận tâm và vô cùng chính xác.
Nhiệm vụ của bạn là giải đáp thắc mắc người dùng TRÊN CƠ SỞ DUY NHẤT là các tài liệu pháp lý được cung cấp ở phần <context>.

<quy_tac_nghiem_ngat>
1. CHỐNG ẢO GIÁC: TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài, không tự suy diễn và không bịa đặt thông tin.
2. XỬ LÝ NGOẠI LỆ: Nếu thông tin trong <context> rỗng, không liên quan hoặc không đủ để trả lời, bạn BẮT BUỘC phải trả lời: "Không đủ thông tin pháp lý để trả lời câu hỏi này."
3. ĐỊNH DẠNG: Trình bày câu trả lời rõ ràng, mạch lạc, sử dụng gạch đầu dòng để phân đoạn.
4. TRÍCH DẪN BẮT BUỘC: Mọi luận điểm phải kèm trích dẫn nguồn. Đặt ở cuối đoạn văn: [ID: <mã_cid>].
</quy_tac_nghiem_ngat>"""


class GeneratorClient:
    """Direct Gemini API caller that mirrors the backend's generator logic."""

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_GENERATOR_MODEL,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        self.rate_limiter = rate_limiter
        self.gen_config = genai.types.GenerationConfig(
            temperature=0.0,
            top_p=0.1,
            top_k=1,
            max_output_tokens=1024,
        )

    async def generate(self, question: str, context_texts: List[str], context_cids: List[int]) -> GenerationResult:
        """Generate an answer given a question and retrieved context."""
        if not context_texts:
            return GenerationResult(
                answer="Không đủ thông tin pháp lý để trả lời câu hỏi này.",
                latency_ms=0.0,
            )

        # Build context block (mirrors LegalPromptBuilder.format_context)
        ctx_parts = []
        for i, text in enumerate(context_texts, 1):
            cid = context_cids[i - 1] if i - 1 < len(context_cids) else -1
            ctx_parts.append(f"\n[Tài liệu {i} - ID: {cid}]:\n{text}\n")
            ctx_parts.append("-" * 40)
        context_str = "".join(ctx_parts)

        user_msg = f"{question}\n\n<context>\n{context_str}\n</context>"

        start = time.perf_counter()
        if self.rate_limiter:
            await self.rate_limiter.wait_turn()

        try:
            response = await asyncio.wait_for(
                self.model.generate_content_async(
                    [{"role": "user", "parts": [user_msg]}],
                    generation_config=self.gen_config,
                ),
                timeout=30.0,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            raise RuntimeError(f"Generation failed after {elapsed:.0f}ms: {e}") from e

        elapsed = (time.perf_counter() - start) * 1000
        answer_text = (response.text or "").strip()
        if not answer_text:
            raise RuntimeError(f"Generation returned empty text after {elapsed:.0f}ms")

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata"):
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count

        return GenerationResult(
            answer=answer_text,
            latency_ms=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: LLM-as-a-Judge (Cohere Command-R-Plus)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JUDGE_PROMPT = """You are a strict legal QA evaluation judge. You must evaluate how well an AI assistant answered a Vietnamese legal question.

You will be given:
1. The original QUESTION
2. The GROUND TRUTH context (the correct legal text)
3. The RETRIEVED CONTEXT (what the system actually found)
4. The AI ANSWER (what the system generated)

Score each dimension on a 1-5 scale (1=terrible, 5=perfect):

**faithfulness**: Does the answer ONLY use information from the retrieved context? (5 = no hallucination at all, 1 = many fabricated claims)
**answer_relevancy**: Does the answer directly address the question? (5 = perfectly on-topic, 1 = completely off-topic)
**completeness**: Does the answer cover all key legal points from the ground truth? (5 = all points covered, 1 = most points missed)
**citation_accuracy**: Are legal references and citations correct and present? (5 = perfect citations, 1 = no or wrong citations)

Return ONLY a compact JSON object:
{{
    "faithfulness": <1-5>,
    "answer_relevancy": <1-5>,
    "completeness": <1-5>,
    "citation_accuracy": <1-5>,
    "reasoning": "<1-2 sentence explanation of your scores>"
}}

Output JSON only, no markdown, no extra text.

---

QUESTION:
{question}

GROUND TRUTH CONTEXT:
{ground_truth}

RETRIEVED CONTEXT:
{retrieved_context}

AI ANSWER:
{answer}
"""


class LLMJudge:
    """Uses Cohere Command-R-Plus as an impartial judge to score RAG outputs."""

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_JUDGE_MODEL,
        rate_limiter: Optional[AsyncRateLimiter] = None,
    ):
        self.client = cohere.AsyncClientV2(api_key=api_key)
        self.model_name = model_name
        self.rate_limiter = rate_limiter

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Unicode-aware tokenization with lightweight stopword filtering."""
        if not text:
            return []
        stopwords = {
            "và", "là", "của", "có", "được", "theo", "tại", "cho", "khi", "để", "với",
            "các", "những", "một", "này", "đó", "trong", "trên", "về", "hoặc", "điều",
            "khoản", "điểm", "luật", "nghị", "định", "thông", "tư", "số",
        }
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        filtered = [t for t in tokens if len(t) > 1 and t not in stopwords]
        return filtered or tokens

    @staticmethod
    def _sanitize_score(value: Any) -> float:
        """Normalize judge score to [1, 5], keep 0 for missing/invalid."""
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score <= 0:
            return 0.0
        if not (score == score):
            return 0.0
        return max(JUDGE_SCORE_MIN, min(JUDGE_SCORE_MAX, score))

    @staticmethod
    def _compute_context_hit(gt_context: str, retrieved_texts: List[str], threshold: float = 0.3) -> bool:
        """
        Check if the ground-truth context appears (partially) in Retrieved Texts.
        Uses token overlap ratio instead of exact string match.
        """
        if not gt_context or not retrieved_texts:
            return False

        gt_tokens = set(LLMJudge._tokenize(gt_context))
        if not gt_tokens:
            return False

        combined_retrieved = " ".join(retrieved_texts)
        retrieved_tokens = set(LLMJudge._tokenize(combined_retrieved))
        if not retrieved_tokens:
            return False

        overlap = len(gt_tokens & retrieved_tokens)
        ratio = overlap / len(gt_tokens)
        adaptive_threshold = 0.45 if len(gt_tokens) > 30 else threshold
        return ratio >= adaptive_threshold

    @staticmethod
    def _compute_cid_recall(gt_cids: List[int], retrieved_cids: List[int]) -> float:
        """Fraction of ground-truth CIDs that appear in retrieved results."""
        if not gt_cids:
            return 0.0
        gt_set = set(gt_cids)
        retrieved_set = set(retrieved_cids)
        found = sum(1 for c in gt_set if c in retrieved_set)
        return found / len(gt_set)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from potentially noisy LLM output."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"No valid JSON in judge response: {text[:200]}")

    async def evaluate(self, record: EvalRecord) -> EvalScores:
        """Score a single evaluation record using the LLM judge."""
        sample = record.sample
        retrieval = record.retrieval
        generation = record.generation

        scores = EvalScores()

        # ── Retrieval metrics (computed deterministically, no LLM needed) ──
        if retrieval and retrieval.source == "backend":
            scores.retrieval_metrics_available = True
            scores.context_hit = self._compute_context_hit(
                sample.ground_truth_context, retrieval.retrieved_texts
            )
            scores.context_cid_recall = self._compute_cid_recall(
                sample.ground_truth_cids, retrieval.retrieved_cids
            )

        # ── Generation metrics (LLM judge) ──
        if generation and generation.answer:
            retrieved_context_preview = ""
            if retrieval and retrieval.retrieved_texts:
                retrieved_context_preview = "\n".join(retrieval.retrieved_texts[:3])

            prompt = JUDGE_PROMPT.format(
                question=sample.question[:PROMPT_QUESTION_CHAR_LIMIT],
                ground_truth=sample.ground_truth_context[:PROMPT_CONTEXT_CHAR_LIMIT],
                retrieved_context=retrieved_context_preview[:PROMPT_CONTEXT_CHAR_LIMIT],
                answer=generation.answer[:PROMPT_ANSWER_CHAR_LIMIT],
            )

            if self.rate_limiter:
                await self.rate_limiter.wait_turn()

            try:
                response = await asyncio.wait_for(
                    self.client.chat(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=400,
                    ),
                    timeout=30.0,
                )
                data = self._extract_json(response.message.content[0].text or "")
            except Exception as e:
                raise RuntimeError(f"Judge failed for qid={sample.qid}: {e}") from e

            scores.faithfulness = self._sanitize_score(data.get("faithfulness", 0))
            scores.answer_relevancy = self._sanitize_score(data.get("answer_relevancy", 0))
            scores.completeness = self._sanitize_score(data.get("completeness", 0))
            scores.citation_accuracy = self._sanitize_score(data.get("citation_accuracy", 0))
            scores.judge_reasoning = str(data.get("reasoning", ""))

        # ── Composite score ──
        gen_scores = [scores.faithfulness, scores.answer_relevancy,
                      scores.completeness, scores.citation_accuracy]
        valid_scores = [s for s in gen_scores if s > 0]
        if valid_scores:
            scores.overall_score = sum(valid_scores) / len(valid_scores)

        return scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retry Utility
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _retry_async(
    fn,
    *args,
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY,
    **kwargs,
):
    """
    Retry an async callable with exponential backoff.
    Prevents data loss from transient API failures (503, timeout, rate-limit).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            log.warning(
                f"  ⟳ Retry {attempt}/{max_attempts} in {delay:.1f}s — "
                f"{type(e).__name__}: {e}"
            )
            await asyncio.sleep(delay)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Orchestrator — Ties everything together
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EvaluationPipeline:
    """
    Orchestrates: Golden Dataset → Retrieval → Generation → LLM Judging → Report.
    Uses asyncio.Semaphore for concurrent execution within rate limits.
    """

    def __init__(
        self,
        gemini_key: str,
        cohere_key: str,
        csv_path: Path = DEFAULT_TRAIN_CSV,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        n_samples: int = DEFAULT_N_SAMPLES,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
        backend_url: str = "http://localhost:8000",
        use_backend: bool = True,
        top_k: int = DEFAULT_TOP_K,
        seed: int = 42,
        mlflow_run_name: Optional[str] = None,
        mlflow_experiment: str = "legal-rag-evaluation",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store config for MLflow logging
        self._n_samples = n_samples
        self._judge_model = judge_model
        self._generator_model = generator_model
        self._top_k = top_k
        self._seed = seed
        self._use_backend = use_backend
        self._mlflow_run_name = mlflow_run_name
        self._mlflow_experiment = mlflow_experiment

        self.gemini_rate_limiter = AsyncRateLimiter(GEMINI_DELAY_SEC)
        self.cohere_rate_limiter = AsyncRateLimiter(COHERE_DELAY_SEC)

        self.dataset_builder = GoldenDatasetBuilder(csv_path, n_samples, seed)
        self.retriever = RetrieverClient(backend_url, top_k, use_backend)
        self.generator = GeneratorClient(
            gemini_key,
            generator_model,
            rate_limiter=self.gemini_rate_limiter,
        )
        self.judge = LLMJudge(
            cohere_key,
            judge_model,
            rate_limiter=self.cohere_rate_limiter,
        )
        self.records: List[EvalRecord] = []

    # ──────────────────────────────────────────────────────────────────────
    # Concurrent inference helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _infer_one(
        self,
        sample: GoldenSample,
        sem: asyncio.Semaphore,
        pbar: tqdm,
    ) -> EvalRecord:
        """Run retrieval + generation for a single sample, respecting the semaphore."""
        async with sem:
            record = EvalRecord(sample=sample)
            try:
                # Retrieval (with retry)
                record.retrieval = await _retry_async(
                    self.retriever.retrieve,
                    sample.question,
                    gt_context=sample.ground_truth_context,
                    gt_cids=sample.ground_truth_cids,
                )

                # Generation (with retry)
                ctx_texts = record.retrieval.retrieved_texts if record.retrieval else []
                ctx_cids = record.retrieval.retrieved_cids if record.retrieval else []
                record.generation = await _retry_async(
                    self.generator.generate,
                    sample.question, ctx_texts, ctx_cids,
                )

            except Exception as e:
                if not record.retrieval:
                    record.retrieval = RetrievalResult([], [], [], 0.0, source="backend_error")
                record.error = str(e)
                log.warning(f"Inference failed for qid={sample.qid}: {e}")

            pbar.update(1)
            return record

    async def _judge_one(
        self,
        record: EvalRecord,
        sem: asyncio.Semaphore,
        pbar: tqdm,
    ) -> EvalRecord:
        """Score a single record via LLM judge, respecting the semaphore."""
        async with sem:
            if record.error:
                record.scores = EvalScores()
            else:
                try:
                    record.scores = await _retry_async(
                        self.judge.evaluate, record,
                    )
                except Exception as e:
                    record.scores = EvalScores()
                    record.error = record.error or str(e)
                    log.warning(f"Judge error for qid={record.sample.qid}: {e}")

            pbar.update(1)
            return record

    # ──────────────────────────────────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Execute the full evaluation pipeline with concurrent API calls."""
        t_start = time.time()
        sem = asyncio.Semaphore(GEMINI_CONCURRENT_LIMIT)

        # ── Step 1: Build Golden Dataset ──
        log.info("=" * 70)
        log.info("STEP 1/4 — Building Golden Dataset")
        samples = self.dataset_builder.build()

        # ── Step 1.5: Check backend connectivity ──
        if not self.retriever.use_backend:
            self.retriever._backend_alive = False
            log.info("⚠️  Backend retrieval is DISABLED (--no_backend) → oracle GT context mode")
        else:
            backend_alive = await self.retriever._check_backend()
            if backend_alive:
                log.info("✅ Backend is ONLINE → Real retrieval will be used")
            else:
                log.info("⚠️  Backend is OFFLINE → Using ground-truth context for generation eval")
                log.info("   (Start the backend to also evaluate retrieval quality)")

        # ── Step 2: Concurrent Inference (Retrieval + Generation) ──
        log.info("=" * 70)
        log.info(f"STEP 2/4 — Running Inference (concurrency={GEMINI_CONCURRENT_LIMIT}, retry={RETRY_MAX_ATTEMPTS})")

        pbar_infer = tqdm(total=len(samples), desc="Inference", unit="q", colour="cyan", file=sys.stdout)
        inference_tasks = [
            self._infer_one(sample, sem, pbar_infer) for sample in samples
        ]
        self.records = await asyncio.gather(*inference_tasks)
        pbar_infer.close()

        # ── Step 3: Concurrent LLM-as-a-Judge Scoring ──
        log.info("=" * 70)
        log.info(f"STEP 3/4 — LLM-as-a-Judge Evaluation (concurrency={GEMINI_CONCURRENT_LIMIT})")

        pbar_judge = tqdm(total=len(self.records), desc="Judging", unit="q", colour="yellow", file=sys.stdout)
        judge_tasks = [
            self._judge_one(record, sem, pbar_judge) for record in self.records
        ]
        self.records = await asyncio.gather(*judge_tasks)
        pbar_judge.close()

        # ── Step 4: Generate Report ──
        log.info("=" * 70)
        log.info("STEP 4/4 — Generating Evaluation Report")
        self._generate_report()
        self._export_detailed_results()

        elapsed = time.time() - t_start
        log.info(f"✅ Evaluation pipeline completed in {elapsed / 60:.1f} minutes")

        # ── Step 5: Log to MLflow (optional) ──
        self._log_to_mlflow(elapsed)

    # ──────────────────────────────────────────────────────────────────────
    # Report Generation
    # ──────────────────────────────────────────────────────────────────────

    def _generate_report(self) -> None:
        """Generate a comprehensive markdown evaluation report."""
        scored = [r for r in self.records if r.scores]
        n_total = len(self.records)
        n_errors = sum(1 for r in self.records if r.error)
        n_scored = len(scored)

        if not scored:
            log.error("No records to report!")
            return

        # ── Aggregate Retrieval Metrics ──
        retrieval_scored = [r for r in scored if r.scores.retrieval_metrics_available]
        n_retrieval_scored = len(retrieval_scored)
        context_hits = sum(1 for r in retrieval_scored if r.scores.context_hit)
        avg_cid_recall = (
            sum(r.scores.context_cid_recall for r in retrieval_scored) / n_retrieval_scored
            if n_retrieval_scored else 0.0
        )

        retrieval_mode_counts: Dict[str, int] = {"backend": 0, "oracle_gt": 0, "backend_error": 0, "missing": 0}
        for r in scored:
            if not r.retrieval:
                retrieval_mode_counts["missing"] += 1
            else:
                retrieval_mode_counts[r.retrieval.source] = retrieval_mode_counts.get(r.retrieval.source, 0) + 1

        # ── Aggregate Generation Metrics ──
        faith_scores = [r.scores.faithfulness for r in scored if r.scores.faithfulness > 0]
        relev_scores = [r.scores.answer_relevancy for r in scored if r.scores.answer_relevancy > 0]
        comp_scores = [r.scores.completeness for r in scored if r.scores.completeness > 0]
        cite_scores = [r.scores.citation_accuracy for r in scored if r.scores.citation_accuracy > 0]
        overall_scores = [r.scores.overall_score for r in scored if r.scores.overall_score > 0]

        def _avg(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        def _make_bar(score: float, max_score: float = 5.0, width: int = 20) -> str:
            filled = int((score / max_score) * width)
            return "█" * filled + "░" * (width - filled)

        # ── Latency Stats ──
        gen_latencies = [r.generation.latency_ms for r in scored if r.generation]
        ret_latencies = [
            r.retrieval.latency_ms for r in retrieval_scored
            if r.retrieval and r.retrieval.latency_ms > 0
        ]

        # ── Score Distributions ──
        def _distribution(scores: List[float]) -> Dict[str, int]:
            buckets = {"1 (Poor)": 0, "2 (Fair)": 0, "3 (Good)": 0, "4 (Very Good)": 0, "5 (Excellent)": 0}
            for s in scores:
                if s <= 1.5:
                    buckets["1 (Poor)"] += 1
                elif s <= 2.5:
                    buckets["2 (Fair)"] += 1
                elif s <= 3.5:
                    buckets["3 (Good)"] += 1
                elif s <= 4.5:
                    buckets["4 (Very Good)"] += 1
                else:
                    buckets["5 (Excellent)"] += 1
            return buckets

        # ── Build the Report ──
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines = [
            f"# 📊 Legal RAG Evaluation Report",
            f"",
            f"**Generated**: {timestamp}",
            f"**Samples Evaluated**: {n_scored:,} / {n_total:,} (errors: {n_errors})",
            f"",
            f"---",
            f"",
            f"## 🏆 Overall Score: {_avg(overall_scores):.2f} / 5.00",
            f"",
            f"```",
            f"  Overall     {_make_bar(_avg(overall_scores))} {_avg(overall_scores):.2f}/5",
            f"```",
            f"",
            f"---",
            f"",
            f"## 📡 Retrieval Quality",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            (
                f"| Retrieval Mode Breakdown | "
                f"backend={retrieval_mode_counts.get('backend', 0)}, "
                f"oracle_gt={retrieval_mode_counts.get('oracle_gt', 0)}, "
                f"backend_error={retrieval_mode_counts.get('backend_error', 0)}, "
                f"missing={retrieval_mode_counts.get('missing', 0)} |"
            ),
            (
                f"| Context Hit Rate | {context_hits}/{n_retrieval_scored} "
                f"({(context_hits / n_retrieval_scored * 100):.1f}%) |"
                if n_retrieval_scored
                else "| Context Hit Rate | N/A (no real backend retrieval in this run) |"
            ),
            (
                f"| CID Recall | {avg_cid_recall:.2%} |"
                if n_retrieval_scored
                else "| CID Recall | N/A (no real backend retrieval in this run) |"
            ),
            f"| Avg Retrieval Latency | {_avg(ret_latencies):.0f} ms |" if ret_latencies else "",
            f"",
            f"---",
            f"",
            f"## 🤖 Generation Quality (LLM-as-a-Judge)",
            f"",
            f"| Metric | Score | Visual |",
            f"|--------|-------|--------|",
            f"| Faithfulness | {_avg(faith_scores):.2f}/5 | {_make_bar(_avg(faith_scores))} |",
            f"| Answer Relevancy | {_avg(relev_scores):.2f}/5 | {_make_bar(_avg(relev_scores))} |",
            f"| Completeness | {_avg(comp_scores):.2f}/5 | {_make_bar(_avg(comp_scores))} |",
            f"| Citation Accuracy | {_avg(cite_scores):.2f}/5 | {_make_bar(_avg(cite_scores))} |",
            f"",
            f"> **Faithfulness** = Chống ảo giác (Hallucination Guard)",
            f"> **Answer Relevancy** = Trả lời đúng trọng tâm",
            f"> **Completeness** = Đầy đủ các điểm pháp lý",
            f"> **Citation Accuracy** = Trích dẫn chính xác",
            f"",
            f"---",
            f"",
            f"## 📊 Score Distribution (Faithfulness)",
            f"",
            f"| Rating | Count |",
            f"|--------|-------|",
        ]

        for rating, count in _distribution(faith_scores).items():
            pct = (count / len(faith_scores) * 100) if faith_scores else 0
            report_lines.append(f"| {rating} | {count} ({pct:.0f}%) |")

        report_lines.extend([
            f"",
            f"---",
            f"",
            f"## ⚡ Performance",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Avg Generation Latency | {_avg(gen_latencies):.0f} ms |" if gen_latencies else "",
            f"| Avg Tokens/Response | {sum(r.generation.completion_tokens for r in scored if r.generation) / n_scored:.0f} |" if n_scored else "",
            f"",
            f"---",
            f"",
            f"## ❌ Worst Performing Samples",
            f"",
        ])

        # Show bottom 5 samples
        worst = sorted(
            [r for r in scored if r.scores.overall_score > 0],
            key=lambda r: r.scores.overall_score
        )[:5]

        for i, r in enumerate(worst, 1):
            report_lines.extend([
                f"### {i}. QID: {r.sample.qid} — Score: {r.scores.overall_score:.2f}/5",
                f"**Q:** {r.sample.question[:150]}...",
                f"**Judge:** {r.scores.judge_reasoning[:200]}",
                f"",
            ])

        report_lines.extend([
            f"---",
            f"",
            f"## ✅ Best Performing Samples",
            f"",
        ])

        best = sorted(
            [r for r in scored if r.scores.overall_score > 0],
            key=lambda r: r.scores.overall_score,
            reverse=True
        )[:5]

        for i, r in enumerate(best, 1):
            report_lines.extend([
                f"### {i}. QID: {r.sample.qid} — Score: {r.scores.overall_score:.2f}/5",
                f"**Q:** {r.sample.question[:150]}...",
                f"**Judge:** {r.scores.judge_reasoning[:200]}",
                f"",
            ])

        # Write report
        report_path = self.output_dir / "evaluation_report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        log.info(f"📝 Report saved: {report_path}")

        # Print summary to console
        log.info("")
        log.info("╔══════════════════════════════════════════════════════════╗")
        log.info("║              EVALUATION SUMMARY                        ║")
        log.info("╠══════════════════════════════════════════════════════════╣")
        log.info(f"║  Overall Score     : {_avg(overall_scores):.2f} / 5.00                     ║")
        log.info(f"║  Faithfulness      : {_avg(faith_scores):.2f} / 5.00                     ║")
        log.info(f"║  Answer Relevancy  : {_avg(relev_scores):.2f} / 5.00                     ║")
        log.info(f"║  Completeness      : {_avg(comp_scores):.2f} / 5.00                     ║")
        log.info(f"║  Citation Accuracy : {_avg(cite_scores):.2f} / 5.00                     ║")
        if n_retrieval_scored:
            context_hit_rate_display = f"{context_hits / n_retrieval_scored * 100:.1f}%"
        else:
            context_hit_rate_display = "N/A"
        log.info(f"║  Context Hit Rate  : {context_hit_rate_display:<30}║")
        log.info("╚══════════════════════════════════════════════════════════╝")

    def _export_detailed_results(self) -> None:
        """Export per-sample results as JSONL for deeper analysis."""
        jsonl_path = self.output_dir / "detailed_results.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in self.records:
                row = {
                    "qid": record.sample.qid,
                    "question": record.sample.question,
                    "ground_truth_cids": record.sample.ground_truth_cids,
                    "ground_truth_context_preview": record.sample.ground_truth_context[:300],
                }
                if record.retrieval:
                    row["retrieval"] = {
                        "n_docs": len(record.retrieval.retrieved_texts),
                        "cids": record.retrieval.retrieved_cids,
                        "latency_ms": record.retrieval.latency_ms,
                        "source": record.retrieval.source,
                    }
                if record.generation:
                    row["generation"] = {
                        "answer_preview": record.generation.answer[:500],
                        "latency_ms": record.generation.latency_ms,
                        "tokens": record.generation.completion_tokens,
                    }
                if record.scores:
                    row["scores"] = {
                        "retrieval_metrics_available": record.scores.retrieval_metrics_available,
                        "context_hit": record.scores.context_hit,
                        "context_cid_recall": record.scores.context_cid_recall,
                        "faithfulness": record.scores.faithfulness,
                        "answer_relevancy": record.scores.answer_relevancy,
                        "completeness": record.scores.completeness,
                        "citation_accuracy": record.scores.citation_accuracy,
                        "overall_score": record.scores.overall_score,
                        "judge_reasoning": record.scores.judge_reasoning,
                    }
                if record.error:
                    row["error"] = record.error

                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        log.info(f"📊 Detailed results: {jsonl_path}")

        # Also export a summary CSV for quick analysis
        csv_path = self.output_dir / "scores_summary.csv"
        rows = []
        for r in self.records:
            if r.scores:
                rows.append({
                    "qid": r.sample.qid,
                    "question": r.sample.question[:100],
                    "retrieval_source": r.retrieval.source if r.retrieval else "missing",
                    "retrieval_metrics_available": r.scores.retrieval_metrics_available,
                    "context_hit": r.scores.context_hit,
                    "cid_recall": round(r.scores.context_cid_recall, 3),
                    "faithfulness": r.scores.faithfulness,
                    "relevancy": r.scores.answer_relevancy,
                    "completeness": r.scores.completeness,
                    "citation": r.scores.citation_accuracy,
                    "overall": round(r.scores.overall_score, 3),
                })
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
            log.info(f"📈 Summary CSV: {csv_path}")

    # ──────────────────────────────────────────────────────────────────────
    # MLflow Tracking (optional)
    # ──────────────────────────────────────────────────────────────────────

    def _log_to_mlflow(self, elapsed_sec: float) -> None:
        """
        Log evaluation hyperparameters, metrics, and output artifacts to MLflow.

        Safe no-op if MLflow is not installed or if --mlflow_run_name was not
        provided.  All errors are caught so a failed MLflow call never crashes
        the evaluation pipeline.
        """
        if not _HAS_MLFLOW or not self._mlflow_run_name:
            if self._mlflow_run_name and not _HAS_MLFLOW:
                log.warning(
                    "[mlflow] --mlflow_run_name was set but mlflow is not installed. "
                    "Install it with: pip install mlflow"
                )
            return

        try:
            mlflow.set_experiment(self._mlflow_experiment)

            scored = [r for r in self.records if r.scores]
            if not scored:
                log.warning("[mlflow] No scored records — skipping MLflow log")
                return

            # ── Compute aggregate metrics ──
            def _avg(vals: list) -> float:
                return sum(vals) / len(vals) if vals else 0.0

            faith_scores   = [r.scores.faithfulness     for r in scored if r.scores.faithfulness > 0]
            relev_scores   = [r.scores.answer_relevancy for r in scored if r.scores.answer_relevancy > 0]
            comp_scores    = [r.scores.completeness     for r in scored if r.scores.completeness > 0]
            cite_scores    = [r.scores.citation_accuracy for r in scored if r.scores.citation_accuracy > 0]
            overall_scores = [r.scores.overall_score    for r in scored if r.scores.overall_score > 0]

            retrieval_scored = [r for r in scored if r.scores.retrieval_metrics_available]
            n_retrieval_scored = len(retrieval_scored)
            context_hits = sum(1 for r in retrieval_scored if r.scores.context_hit)
            context_hit_rate = (context_hits / n_retrieval_scored) if n_retrieval_scored else 0.0
            avg_cid_recall = _avg([r.scores.context_cid_recall for r in retrieval_scored])

            gen_latencies = [r.generation.latency_ms for r in scored if r.generation]
            n_errors = sum(1 for r in self.records if r.error)

            with mlflow.start_run(run_name=self._mlflow_run_name):
                # ── Hyperparameters ──
                mlflow.log_params({
                    "n_samples":       self._n_samples,
                    "top_k":           self._top_k,
                    "judge_model":     self._judge_model,
                    "generator_model": self._generator_model,
                    "seed":            self._seed,
                    "use_backend":     self._use_backend,
                })

                # ── Generation Metrics ──
                mlflow.log_metrics({
                    "faithfulness":      _avg(faith_scores),
                    "answer_relevancy":  _avg(relev_scores),
                    "completeness":      _avg(comp_scores),
                    "citation_accuracy": _avg(cite_scores),
                    "overall_score":     _avg(overall_scores),
                })

                # ── Retrieval Metrics ──
                mlflow.log_metrics({
                    "context_hit_rate":  context_hit_rate,
                    "cid_recall":        avg_cid_recall,
                })

                # ── Operational Metrics ──
                mlflow.log_metrics({
                    "avg_gen_latency_ms":  _avg(gen_latencies),
                    "n_errors":            float(n_errors),
                    "elapsed_minutes":     elapsed_sec / 60.0,
                })

                # ── Artifacts ──
                report_path = self.output_dir / "evaluation_report.md"
                jsonl_path  = self.output_dir / "detailed_results.jsonl"
                csv_path    = self.output_dir / "scores_summary.csv"

                for artifact in (report_path, jsonl_path, csv_path):
                    if artifact.exists():
                        mlflow.log_artifact(str(artifact))

            log.info("[mlflow] ✅ Run '%s' logged to experiment '%s'",
                     self._mlflow_run_name, self._mlflow_experiment)

        except Exception as exc:
            log.warning("[mlflow] Failed to log run (non-fatal): %s", exc)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Legal RAG Evaluation Pipeline — LLM-as-a-Judge Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gemini_key", type=str, default=os.getenv("GEMINI_API_KEY", ""),
                    help="Gemini API Key (or set GEMINI_API_KEY env var)")
    p.add_argument("--cohere_key", type=str, default=os.getenv("COHERE_API_KEY", ""),
                    help="Cohere API Key (or set COHERE_API_KEY env var)")
    p.add_argument("--train_csv", type=str, default=str(DEFAULT_TRAIN_CSV),
                    help="Path to train.csv golden dataset")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                    help="Output directory for reports")
    p.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES,
                    help="Number of samples to evaluate (1000-2000 recommended)")
    p.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                    help="LLM model for judging")
    p.add_argument("--generator_model", type=str, default=DEFAULT_GENERATOR_MODEL,
                    help="LLM model for answer generation")
    p.add_argument("--backend_url", type=str, default="http://localhost:8000",
                    help="Backend URL for real retrieval")
    p.add_argument("--no_backend", action="store_true",
                    help="Skip backend retrieval; use GT context for generation-only eval")
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                    help="Number of documents to retrieve")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible sampling")
    # ── MLflow tracking ──
    p.add_argument("--mlflow_run_name", type=str, default=None,
                    help="If set, log this run to MLflow under this name. Requires: pip install mlflow")
    p.add_argument("--mlflow_experiment", type=str, default="legal-rag-evaluation",
                    help="MLflow experiment name to group runs under")
    return p.parse_args()


async def main():
    args = parse_args()

    if not args.gemini_key:
        log.error("❌ GEMINI_API_KEY is required. Pass --gemini_key or set GEMINI_API_KEY env var.")
        sys.exit(1)

    if not args.cohere_key:
        log.error("❌ COHERE_API_KEY is required. Pass --cohere_key or set COHERE_API_KEY env var.")
        sys.exit(1)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║     Legal RAG Evaluation Pipeline v1.0                  ║")
    log.info("║     LLM-as-a-Judge with Cohere Command R+               ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"  Config:")
    log.info(f"    Samples    : {args.n_samples}")
    log.info(f"    Judge      : {args.judge_model}")
    log.info(f"    Generator  : {args.generator_model}")
    log.info(f"    Backend    : {'disabled' if args.no_backend else args.backend_url}")
    log.info(f"    Output     : {args.output_dir}")

    pipeline = EvaluationPipeline(
        gemini_key=args.gemini_key,
        cohere_key=args.cohere_key,
        csv_path=Path(args.train_csv),
        output_dir=Path(args.output_dir),
        n_samples=args.n_samples,
        judge_model=args.judge_model,
        generator_model=args.generator_model,
        backend_url=args.backend_url,
        use_backend=not args.no_backend,
        top_k=args.top_k,
        seed=args.seed,
        mlflow_run_name=args.mlflow_run_name,
        mlflow_experiment=args.mlflow_experiment,
    )

    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
