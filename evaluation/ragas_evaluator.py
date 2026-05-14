#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ragas_evaluator.py
==================
RAGAS-based evaluation pipeline for the Vietnamese Legal RAG Chatbot.
Runs alongside the existing LLM-as-a-Judge pipeline (legal_rag_evaluator.py).

Architecture:
  - Generator : Xiaomi MiMo v2.5 Pro  (generates answers)
  - Evaluator : Gemini 3.1 Flash Lite (RAGAS judge)
  - Embeddings: Google text-embedding-004 (for AnswerRelevancy)

Metrics:
  - Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

Usage:
    python ragas_evaluator.py --mimo_key KEY --gemini_key KEY
    python ragas_evaluator.py --n_samples 10 --no_backend

Dependencies:
    pip install ragas langchain-google-genai openai pandas tqdm httpx
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data" / "raw_csv" / "train.csv"
DEFAULT_OUTPUT_DIR = EVAL_DIR / "eval_results" / "ragas"
DEFAULT_N_SAMPLES = 50
DEFAULT_TOP_K = 5

# MiMo (Generator)
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"

# Gemini (RAGAS Evaluator)
GEMINI_EVAL_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# System prompt — mirrors backend generator
SYSTEM_PROMPT = (
    "Bạn là một chuyên gia tư vấn Pháp luật Việt Nam xuất sắc, tận tâm và vô cùng chính xác.\n"
    "Nhiệm vụ của bạn là giải đáp thắc mắc người dùng TRÊN CƠ SỞ DUY NHẤT là các tài liệu pháp lý "
    "được cung cấp ở phần <context>.\n\n"
    "<quy_tac_nghiem_ngat>\n"
    "1. CHỐNG ẢO GIÁC: TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài.\n"
    "2. XỬ LÝ NGOẠI LỆ: Nếu thông tin trong <context> không đủ, trả lời: "
    '"Không đủ thông tin pháp lý để trả lời câu hỏi này."\n'
    "3. ĐỊNH DẠNG: Trình bày rõ ràng, mạch lạc, sử dụng gạch đầu dòng.\n"
    "4. TRÍCH DẪN BẮT BUỘC: Mọi luận điểm kèm trích dẫn nguồn [ID: <mã_cid>].\n"
    "</quy_tac_nghiem_ngat>"
)

RETRY_MAX = 3
RETRY_DELAY = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("RagasEvaluator")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Golden Dataset Builder (same logic as existing evaluator)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_golden_dataset(csv_path: Path, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Load and sample golden dataset from train.csv."""
    log.info(f"📂 Loading golden dataset from: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    df = df.dropna(subset=["question", "context"])
    df = df[df["question"].str.strip().str.len() > 10]
    df = df[df["context"].str.strip().str.len() > 20]
    n = min(n_samples, len(df))
    sampled = df.sample(n=n, random_state=seed)
    log.info(f"   ✅ Sampled {n} rows from {len(df)} valid rows")
    return sampled


def load_golden_jsonl(jsonl_path: Path, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Load silver/golden dataset from JSONL produced by build_golden_dataset.py.

    Maps golden set schema fields to the column names expected by run_pipeline:
      question          → question
      reference_answer  → context (used as oracle GT)
      reference_context → context (joined if list)
      reference_cids    → cid
    """
    import random
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not obj.get("question") or not obj.get("reference_context"):
                continue
            ctx = obj["reference_context"]
            ctx_str = "\n\n".join(ctx) if isinstance(ctx, list) else str(ctx)
            records.append({
                "question": obj["question"],
                "context": ctx_str,
                "cid": str(obj.get("reference_cids", [])),
                "reference_answer": obj.get("reference_answer", ""),
                "intent": obj.get("intent", ""),
                "is_negative": obj.get("is_negative", False),
                "tier": obj.get("tier", ""),
                "qid": 0,
            })
    if not records:
        raise ValueError(f"No valid records found in {jsonl_path}")
    rng = random.Random(seed)
    rng.shuffle(records)
    sampled = records[:n_samples]
    df = pd.DataFrame(sampled)
    log.info("📂 Loaded %d records from JSONL (n_samples=%d, path=%s)", len(df), n_samples, jsonl_path)
    return df



def parse_context(raw: str) -> str:
    """Parse list-string context format from train.csv."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return "\n\n".join(str(item).strip() for item in parsed if str(item).strip())
    except (ValueError, SyntaxError):
        pass
    return raw.strip()


def parse_cids(raw: str) -> List[int]:
    """Parse CID field from train.csv."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [int(c) for c in parsed]
        return [int(parsed)]
    except (ValueError, SyntaxError):
        return [int(n) for n in re.findall(r"\d+", str(raw))]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Retrieval (backend or oracle fallback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_backend(backend_url: str) -> bool:
    """Ping backend to check availability."""
    try:
        for path in ("/health", "/docs"):
            r = httpx.get(f"{backend_url}{path}", timeout=5.0)
            if r.status_code == 200:
                return True
    except Exception:
        pass
    return False


def retrieve_from_backend(question: str, backend_url: str, top_k: int) -> List[str]:
    """Retrieve documents from running backend."""
    r = httpx.post(
        f"{backend_url}/api/retrieve",
        json={"message": question, "top_k": top_k, "include_content": True},
        timeout=60.0,
    )
    r.raise_for_status()
    return [s.get("content", "") for s in r.json().get("sources", [])]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Generation with MiMo v2.5 Pro
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_answer(
    client: OpenAI,
    question: str,
    context_texts: List[str],
    context_cids: List[int],
) -> str:
    """Generate answer using MiMo v2.5 Pro."""
    if not context_texts:
        return "Không đủ thông tin pháp lý để trả lời câu hỏi này."

    ctx_parts = []
    for i, text in enumerate(context_texts, 1):
        cid = context_cids[i - 1] if i - 1 < len(context_cids) else -1
        ctx_parts.append(f"\n[Tài liệu {i} - ID: {cid}]:\n{text}\n" + "-" * 40)
    context_str = "".join(ctx_parts)
    user_msg = f"{question}\n\n<context>\n{context_str}\n</context>"

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = client.chat.completions.create(
                model=MIMO_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            answer = (resp.choices[0].message.content or "").strip()
            if answer:
                return answer
        except Exception as e:
            if attempt == RETRY_MAX:
                log.warning(f"MiMo generation failed after {RETRY_MAX} retries: {e}")
                return ""
            time.sleep(RETRY_DELAY * attempt)
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: RAGAS Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ragas_evaluation(
    samples: List[Dict[str, Any]],
    gemini_key: str,
) -> pd.DataFrame:
    """Run RAGAS evaluation on prepared samples."""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    log.info("🔧 Initializing RAGAS evaluator (Gemini 3.1 Flash Lite)...")

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=GEMINI_EVAL_MODEL,
            google_api_key=gemini_key,
            temperature=0.0,
            max_retries=3,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=gemini_key,
        )
    )

    # Build RAGAS dataset
    ragas_samples = []
    for s in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["retrieved_contexts"],
                reference=s["reference"],
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)
    log.info(f"📊 Running RAGAS evaluation on {len(ragas_samples)} samples...")

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    log.info("✅ RAGAS evaluation complete")
    return result.to_pandas()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Report Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_report(
    results_df: pd.DataFrame,
    samples: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any],
) -> None:
    """Generate markdown report, CSV, and JSONL from RAGAS results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = [c for c in results_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    means = {col: results_df[col].mean() for col in metric_cols if results_df[col].dtype in ("float64", "float32")}

    def bar(score: float, width: int = 20) -> str:
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 📊 RAGAS Evaluation Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Samples**: {len(results_df)}",
        f"**Generator**: {config.get('generator', MIMO_MODEL)}",
        f"**Evaluator**: {config.get('evaluator', GEMINI_EVAL_MODEL)}",
        "",
        "---",
        "",
        "## 🏆 Metric Summary (0.0 — 1.0 scale)",
        "",
        "| Metric | Score | Visual |",
        "|--------|-------|--------|",
    ]

    display_names = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    for col, avg in means.items():
        name = display_names.get(col, col)
        lines.append(f"| {name} | {avg:.4f} | {bar(avg)} |")

    lines.extend([
        "",
        "> **Faithfulness** = Chống ảo giác (Hallucination Guard)",
        "> **Answer Relevancy** = Trả lời đúng trọng tâm",
        "> **Context Precision** = Context truy xuất có liên quan",
        "> **Context Recall** = Context truy xuất đầy đủ thông tin",
        "",
        "---",
        "",
        "## 📉 Worst Performing Samples",
        "",
    ])

    if "faithfulness" in results_df.columns:
        worst = results_df.nsmallest(5, "faithfulness")
        for i, (_, row) in enumerate(worst.iterrows(), 1):
            q = str(row.get("user_input", ""))[:150]
            f_score = row.get("faithfulness", 0)
            lines.append(f"### {i}. Faithfulness={f_score:.3f}")
            lines.append(f"**Q:** {q}...")
            lines.append("")

    # Write report
    report_path = output_dir / "ragas_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"📝 Report: {report_path}")

    # Write CSV
    csv_path = output_dir / "ragas_scores.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"📈 CSV: {csv_path}")

    # Write JSONL (per-sample detail)
    jsonl_path = output_dir / "ragas_detailed.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(results_df.iterrows()):
            record = {
                "qid": samples[idx].get("qid", idx) if idx < len(samples) else idx,
                "question": str(row.get("user_input", ""))[:200],
                "answer_preview": str(row.get("response", ""))[:500],
            }
            for col in metric_cols:
                if col in row:
                    val = row[col]
                    record[col] = float(val) if pd.notna(val) else None
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"📊 JSONL: {jsonl_path}")

    # Console summary
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║              RAGAS EVALUATION SUMMARY                  ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    for col, avg in means.items():
        name = display_names.get(col, col)
        log.info(f"║  {name:<22s}: {avg:.4f}                          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(args: argparse.Namespace) -> None:
    """Orchestrate: load data → retrieve → generate (MiMo) → evaluate (RAGAS) → report."""
    t_start = time.time()

    # Validate keys
    if not args.mimo_key:
        log.error("❌ MIMO_API_KEY required. Pass --mimo_key or set MIMO_API_KEY env var.")
        sys.exit(1)
    if not args.gemini_key:
        log.error("❌ GEMINI_API_KEY required. Pass --gemini_key or set GEMINI_API_KEY env var.")
        sys.exit(1)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║     RAGAS Evaluation Pipeline                          ║")
    log.info("║     Generator : MiMo v2.5 Pro (Xiaomi)                ║")
    log.info("║     Evaluator : Gemini 3.1 Flash Lite (Google)        ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"  Samples : {args.n_samples}")
    log.info(f"  Backend : {'disabled' if args.no_backend else args.backend_url}")
    log.info(f"  Output  : {args.output_dir}")

    # Step 1: Load golden dataset (JSONL takes priority over CSV)
    log.info("=" * 60)
    log.info("STEP 1/4 — Loading Golden Dataset")
    if args.golden_jsonl:
        df = load_golden_jsonl(Path(args.golden_jsonl), args.n_samples, args.seed)
    else:
        df = load_golden_dataset(Path(args.train_csv), args.n_samples, args.seed)

    # Step 2: Check backend
    use_backend = not args.no_backend
    backend_alive = False
    if use_backend:
        backend_alive = check_backend(args.backend_url)
        if backend_alive:
            log.info("✅ Backend ONLINE → real retrieval")
        else:
            log.info("⚠️  Backend OFFLINE → oracle GT context mode")

    # Step 3: Generate answers with MiMo
    log.info("=" * 60)
    log.info("STEP 2/4 — Generating Answers (MiMo v2.5 Pro)")
    mimo_client = OpenAI(api_key=args.mimo_key, base_url=MIMO_BASE_URL)

    samples = []
    pbar = tqdm(df.iterrows(), total=len(df), desc="Generate", unit="q", colour="cyan")
    for _, row in pbar:
        question = row["question"].strip()
        gt_context = parse_context(row["context"])
        gt_cids = parse_cids(str(row.get("cid", "[]")))

        # Retrieval
        if backend_alive:
            try:
                retrieved = retrieve_from_backend(question, args.backend_url, args.top_k)
            except Exception as e:
                log.warning(f"Backend retrieval failed: {e}, using GT fallback")
                retrieved = [gt_context] if gt_context else []
        else:
            retrieved = [gt_context] if gt_context else []

        # Generation
        answer = generate_answer(mimo_client, question, retrieved, gt_cids)

        samples.append({
            "qid": int(row.get("qid", 0) or 0),
            "question": question,
            "answer": answer,
            "retrieved_contexts": retrieved,
            "reference": gt_context,
        })

    log.info(f"   Generated {len(samples)} answers")

    # Step 4: RAGAS evaluation
    log.info("=" * 60)
    log.info("STEP 3/4 — RAGAS Evaluation (Gemini 3.1 Flash Lite)")
    results_df = run_ragas_evaluation(samples, args.gemini_key)

    # Step 5: Report
    log.info("=" * 60)
    log.info("STEP 4/4 — Generating Report")
    output_dir = Path(args.output_dir)
    generate_report(
        results_df,
        samples,
        output_dir,
        config={
            "generator": MIMO_MODEL,
            "evaluator": GEMINI_EVAL_MODEL,
            "n_samples": args.n_samples,
            "backend": "real" if backend_alive else "oracle_gt",
        },
    )

    elapsed = time.time() - t_start
    log.info(f"✅ Pipeline completed in {elapsed / 60:.1f} minutes")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAGAS Evaluation Pipeline — Legal RAG Chatbot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mimo_key", type=str, default=os.getenv("MIMO_API_KEY", ""),
                    help="Xiaomi MiMo API Key")
    p.add_argument("--gemini_key", type=str, default=os.getenv("GEMINI_API_KEY", ""),
                    help="Google Gemini API Key")
    p.add_argument("--train_csv", type=str, default=str(DEFAULT_TRAIN_CSV))
    p.add_argument("--golden_jsonl", type=str, default=None,
                    help="Path to JSONL file from build_golden_dataset.py (takes priority over --train_csv)")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--backend_url", type=str, default="http://localhost:8000")
    p.add_argument("--no_backend", action="store_true",
                    help="Skip backend retrieval; use GT context")
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



if __name__ == "__main__":
    run_pipeline(parse_args())
