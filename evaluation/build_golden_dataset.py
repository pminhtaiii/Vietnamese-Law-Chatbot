"""
build_golden_dataset.py — Build a golden evaluation dataset from Qdrant
=======================================================================

Sample chunks from the `legal_docs` collection with stratified sampling by
legal_type, deduplicate by doc_id, and optionally generate questions/answers
using an LLM. Output is JSONL matching the golden set schema.

Usage:
  # Sample chunks only (no LLM — produces stub records for manual review)
  python evaluation/build_golden_dataset.py

  # Generate questions/answers via OpenAI-compatible API
  python evaluation/build_golden_dataset.py --generate-questions \\
      --api-key sk-... --model gpt-4o-mini

  # Custom Qdrant URL and collection
  python evaluation/build_golden_dataset.py --qdrant-url http://myhost:6333 \\
      --collection legal_docs

  # Reproducible sampling with fixed seed
  python evaluation/build_golden_dataset.py --seed 42

Output schema (one JSON object per line):
  {
    "question": "...",
    "reference_answer": "...",
    "reference_context": ["..."],
    "reference_cids": [12345]
  }
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "legal_docs"
DEFAULT_TARGET = 100
DEFAULT_MIN_TEXT_LEN = 100
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 5
DEFAULT_CONCURRENCY = 3
DEFAULT_MAX_RETRIES = 3
DEFAULT_NEG_RATIO = 0.10  # ~10% unanswerable (negative) samples

# MiMo API (Xiaomi) — default provider
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_GEN_MODEL = "mimo-v2.5"        # generator: lighter, faster
MIMO_CRITIC_MODEL = "mimo-v2.5-pro"  # critic: stronger, validates generator

# Keep OpenAI-compatible fallback constants
DEFAULT_MODEL = MIMO_GEN_MODEL
DEFAULT_BASE_URL = MIMO_BASE_URL
DEFAULT_CRITIC_MODEL = MIMO_CRITIC_MODEL

PAYLOAD_LEGAL_TYPE = "loai_van_ban"
PAYLOAD_DOC_ID = "doc_id"
PAYLOAD_TEXT = "text"

OUTPUT_DIR = Path(__file__).resolve().parent / "eval_results"

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("golden_builder")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_legal_types(client: QdrantClient, collection: str) -> List[str]:
    """Get all distinct legal_type values from the collection."""
    types: set[str] = set()

    result, _ = client.scroll(
        collection_name=collection,
        limit=10000,
        with_payload=[PAYLOAD_LEGAL_TYPE],
        with_vectors=False,
    )
    for point in result:
        lt = point.payload.get(PAYLOAD_LEGAL_TYPE, "")
        if lt:
            types.add(str(lt))

    if not types:
        log.warning("No legal_type values found in collection '%s'", collection)
    return sorted(types)


def chunk_statistics(samples: List[Dict[str, Any]]) -> dict:
    """Compute summary statistics for a list of sampled chunks."""
    lengths = [len(s["text"]) for s in samples]
    doc_ids = [s["doc_id"] for s in samples]
    return {
        "total_chunks": len(samples),
        "unique_docs": len(set(doc_ids)),
        "text_len_min": min(lengths) if lengths else 0,
        "text_len_max": max(lengths) if lengths else 0,
        "text_len_mean": sum(lengths) / len(lengths) if lengths else 0,
        "legal_type_distribution": dict(Counter(s["legal_type"] for s in samples)),
    }


# ─── SAMPLING ─────────────────────────────────────────────────────────────────

def sample_chunks(
    client: QdrantClient,
    collection: str,
    target: int = DEFAULT_TARGET,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Stratified sampling of chunks from Qdrant.

    1. Get distinct legal_type values.
    2. For each type, scroll a proportional share (minimum 3 per type).
    3. Deduplicate by doc_id — skip chunks from documents already sampled.
    4. Filter out chunks with < min_text_len characters.
    """
    legal_types = get_legal_types(client, collection)
    if not legal_types:
        log.error("Cannot proceed: no legal_type values found.")
        return []

    n_types = len(legal_types)
    per_type = max(3, target // n_types)
    log.info(
        "Sampling %d chunks: %d types × ~%d per type (target=%d)",
        target, n_types, per_type, target,
    )

    all_samples: List[Dict[str, Any]] = []
    seen_doc_ids: set = set()
    rng = random.Random(seed)

    for lt in legal_types:
        filter_ = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key=PAYLOAD_LEGAL_TYPE,
                    match=qmodels.MatchValue(value=lt),
                )
            ]
        )

        lt_samples: List[Dict[str, Any]] = []
        offset = None
        scan_limit = per_type * 10  # overscan for dedup

        while len(lt_samples) < per_type:
            try:
                result, next_offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=filter_,
                    limit=scan_limit,
                    with_payload=[PAYLOAD_TEXT, PAYLOAD_DOC_ID, PAYLOAD_LEGAL_TYPE],
                    with_vectors=False,
                    offset=offset,
                )
            except Exception as exc:
                log.error("Error scrolling for legal_type='%s': %s", lt, exc)
                break

            for point in result:
                if len(lt_samples) >= per_type:
                    break

                text = point.payload.get(PAYLOAD_TEXT, "")
                doc_id = point.payload.get(PAYLOAD_DOC_ID)

                if not text or len(text) < min_text_len:
                    continue
                if doc_id in seen_doc_ids:
                    continue

                seen_doc_ids.add(doc_id)
                lt_samples.append({
                    "cid": point.id,
                    "text": text,
                    "doc_id": doc_id,
                    "legal_type": lt,
                })

            if next_offset is None:
                break
            offset = next_offset

        all_samples.extend(lt_samples)

    # Fill remaining quota if under-sampled
    if len(all_samples) < target:
        log.info(
            "Under-sampled (%d/%d). Scanning for more unique doc_id chunks...",
            len(all_samples), target,
        )
        remaining_filter = qmodels.Filter(
            must_not=[
                qmodels.FieldCondition(
                    key=PAYLOAD_DOC_ID,
                    match=qmodels.AnyValue(value=list(seen_doc_ids)),
                )
            ]
        )
        try:
            extra_result, _ = client.scroll(
                collection_name=collection,
                scroll_filter=remaining_filter,
                limit=(target - len(all_samples)) * 5,
                with_payload=[PAYLOAD_TEXT, PAYLOAD_DOC_ID, PAYLOAD_LEGAL_TYPE],
                with_vectors=False,
            )
            for point in extra_result:
                if len(all_samples) >= target:
                    break
                text = point.payload.get(PAYLOAD_TEXT, "")
                doc_id = point.payload.get(PAYLOAD_DOC_ID)
                if not text or len(text) < min_text_len:
                    continue
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                all_samples.append({
                    "cid": point.id,
                    "text": text,
                    "doc_id": doc_id,
                    "legal_type": point.payload.get(PAYLOAD_LEGAL_TYPE, "unknown"),
                })
        except Exception as exc:
            log.warning("Fallback scroll failed: %s", exc)

    # Deduplicate (safety net)
    seen_cids: set = set()
    deduped: List[Dict[str, Any]] = []
    for s in all_samples:
        if s["cid"] not in seen_cids:
            seen_cids.add(s["cid"])
            deduped.append(s)
    all_samples = deduped

    if len(all_samples) < target:
        log.warning(
            "Only %d unique doc_id chunks sampled (target was %d).",
            len(all_samples), target,
        )

    rng.shuffle(all_samples)
    stats = chunk_statistics(all_samples)
    log.info("Sampled %d chunks from %d unique documents.", stats["total_chunks"], stats["unique_docs"])
    log.info("Text length: min=%d, max=%d, mean=%d", stats["text_len_min"], stats["text_len_max"], int(stats["text_len_mean"]))
    log.info("Legal type distribution: %s", stats["legal_type_distribution"])

    return all_samples


# ─── LLM QUESTION GENERATION ─────────────────────────────────────────────────

QUESTION_GEN_SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: tạo câu hỏi đánh giá cho hệ thống RAG.

Quy tắc:
1. Viết 1-3 câu hỏi tiếng Việt, tự nhiên như người dùng thật sự hỏi.
2. Câu hỏi phải trả lời trực tiếp được từ nội dung chunk.
3. Tránh câu hỏi mơ tạp hoặc cần kiến thức bên ngoài.
4. Mỗi câu hỏi phải thuộc một trong các intent: LEGAL_LOOKUP (định nghĩa/tra cứu), PROCEDURE (thủ tục), CONDITION (điều kiện/nếu-thì), COMPARE (so sánh).
5. reference_answer phải ngắn gọn, chính xác, dựa trên chunk.
6. Trả về JSON array (không markdown). Mỗi phần tử: {"question": "...", "answer": "...", "intent": "LEGAL_LOOKUP|PROCEDURE|CONDITION|COMPARE"}

Ví dụ output:
[{"question": "Điều kiện để được miễn thuế thu nhập cá nhân là gì?", "answer": "Cá nhân được miễn thuế khi...", "intent": "CONDITION"}]"""

NEGATIVE_GEN_SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: tạo câu hỏi KHÔNG THỂ trả lời từ chunk được cung cấp.

Quy tắc:
1. Câu hỏi phải nghe có vẻ hợp lý và liên quan đến chủ đề pháp lý trong chunk.
2. Nhưng câu hỏi KHÔNG có đáp án trong nội dung chunk (cần kiến thức ngoài, hoặc chunk không đề cập).
3. Chỉ tạo 1 câu hỏi duy nhất.
4. Trả về JSON (không markdown): {"question": "...", "answer": "KHÔNG_CÓ_THÔNG_TIN", "intent": "NEGATIVE"}"""

CRITIC_SYSTEM_PROMPT = """You are a strict quality validator for a Vietnamese legal RAG evaluation dataset.

Given a source chunk and a generated Q&A pair, score each dimension 1-5:
- answerability (1-5): Can the question be fully answered from THIS chunk alone? (5=yes completely, 1=not at all)
- correctness (1-5): Is the reference_answer factually accurate per the chunk? (5=perfectly correct, 1=wrong/hallucinated)
- clarity (1-5): Is the question clear, natural Vietnamese, unambiguous? (5=excellent, 1=confusing)

For NEGATIVE intent questions: answerability should be LOW (1-2) since they are intentionally unanswerable.

Return ONLY compact JSON, no markdown:
{"answerability": N, "correctness": N, "clarity": N, "reasoning": "<1 sentence>"}"""


def make_question_prompt(text: str, legal_type: str) -> str:
    return (
        f"Loại văn bản: {legal_type}\n\n"
        f"Nội dung chunk:\n{text}\n\n"
        f"Hãy tạo 1-3 câu hỏi (với intent label) có thể trả lời trực tiếp từ chunk trên."
    )


def make_negative_prompt(text: str, legal_type: str) -> str:
    return (
        f"Loại văn bản: {legal_type}\n\n"
        f"Nội dung chunk:\n{text}\n\n"
        f"Hãy tạo 1 câu hỏi pháp lý KHÔNG THỂ trả lời từ chunk trên."
    )


def make_critic_prompt(text: str, question: str, answer: str, intent: str) -> str:
    return (
        f"SOURCE CHUNK:\n{text}\n\n"
        f"QUESTION (intent={intent}): {question}\n"
        f"REFERENCE ANSWER: {answer}\n\n"
        f"Validate this Q&A pair against the source chunk."
    )


async def _call_llm(
    client: Any,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[List[Dict[str, str]]]:
    """Call LLM with retry and return parsed question/answer pairs."""
    import openai as openai_lib

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": QUESTION_GEN_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=800,
                )
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                # Remove first and last fence lines
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            questions = json.loads(content)
            if isinstance(questions, list) and len(questions) > 0:
                return questions
        except openai_lib.RateLimitError:
            wait = 2 ** (attempt + 1)
            log.warning("Rate limited. Retrying in %ds...", wait)
            await asyncio.sleep(wait)
        except json.JSONDecodeError:
            log.warning("Invalid JSON from LLM (attempt %d/%d)", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        except Exception as exc:
            log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

    log.error("All %d LLM attempts failed for a chunk.", max_retries)
    return None


async def _call_critic(
    client: Any,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    """Call critic LLM (MiMo v2.5 Pro) and return quality scores dict."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            scores = json.loads(content)
            if all(k in scores for k in ("answerability", "correctness", "clarity")):
                return scores
        except json.JSONDecodeError:
            log.warning("Critic: invalid JSON (attempt %d/%d)", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
        except Exception as exc:
            log.warning("Critic call failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
    log.error("Critic: all %d attempts failed.", max_retries)
    return None


async def validate_samples(
    qa_records: List[Dict[str, Any]],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    critic_model: str = DEFAULT_CRITIC_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run MiMo v2.5 Pro critic on all Q&A records.

    Returns: (kept, flagged, rejected)
      kept     — composite score >=9 (all dims >=3); NEGATIVE with answerability <=2
      flagged  — any score ==2 → needs human review
      rejected — any score ==1
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)

    prompts = [
        make_critic_prompt(r["text"], r["question"], r["reference_answer"], r.get("intent", "UNKNOWN"))
        for r in qa_records
    ]
    results = await asyncio.gather(*[
        _call_critic(client, critic_model, p, semaphore) for p in prompts
    ])

    kept, flagged, rejected = [], [], []
    for rec, scores in zip(qa_records, results):
        if scores is None:
            rec["critic"] = {"error": "critic_failed"}
            rec["tier"] = "flagged"
            flagged.append(rec)
            continue

        rec["critic"] = scores
        a = int(scores.get("answerability", 0))
        c = int(scores.get("correctness", 0))
        cl = int(scores.get("clarity", 0))
        intent = rec.get("intent", "")

        if intent == "NEGATIVE":
            # Unanswerable samples: low answerability is correct
            if a <= 2 and cl >= 3:
                rec["tier"] = "kept"
                kept.append(rec)
            elif cl < 2:
                rec["tier"] = "rejected"
                rejected.append(rec)
            else:
                rec["tier"] = "flagged"
                flagged.append(rec)
        elif min(a, c, cl) == 1:
            rec["tier"] = "rejected"
            rejected.append(rec)
        elif min(a, c, cl) == 2:
            rec["tier"] = "flagged"
            flagged.append(rec)
        else:
            rec["tier"] = "kept"
            kept.append(rec)

    log.info("Critic: %d kept | %d flagged | %d rejected", len(kept), len(flagged), len(rejected))
    return kept, flagged, rejected


async def generate_negative_samples(
    samples: List[Dict[str, Any]],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    neg_count: int = 10,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """Generate intentionally unanswerable Q&A records from a random chunk subset."""
    from openai import AsyncOpenAI
    if not samples:
        return []
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)

    subset = random.sample(samples, min(neg_count, len(samples)))

    async def _gen_one(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = make_negative_prompt(chunk["text"], chunk["legal_type"])
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": NEGATIVE_GEN_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5,
                    max_tokens=300,
                )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            result = json.loads(content)
            if isinstance(result, dict) and result.get("question"):
                return {
                    "cid": chunk["cid"],
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "legal_type": chunk["legal_type"],
                    "question": result["question"],
                    "reference_answer": "KHÔNG_CÓ_THÔNG_TIN",
                    "intent": "NEGATIVE",
                    "is_negative": True,
                }
        except Exception as exc:
            log.warning("Negative gen failed: %s", exc)
        return None

    results = await asyncio.gather(*[_gen_one(c) for c in subset])
    neg_records = [r for r in results if r is not None]
    log.info("Generated %d negative samples", len(neg_records))
    return neg_records


async def generate_questions_for_samples(
    samples: List[Dict[str, Any]],
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """Generate questions/answers for sampled chunks using an OpenAI-compatible LLM."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        log.error("openai package is required: pip install openai")
        return samples

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)

    total_questions = 0
    failed = 0
    t0 = time.time()

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]
        prompts = [
            make_question_prompt(s["text"], s["legal_type"]) for s in batch
        ]

        results = await asyncio.gather(
            *[_call_llm(client, model, p, semaphore) for p in prompts]
        )

        for sample, result in zip(batch, results):
            if result:
                # Store as list — each question becomes a separate JSONL record
                sample["generated_qas"] = result
                total_questions += len(result)
            else:
                sample["generated_qas"] = []
                failed += 1

        processed = min(batch_start + batch_size, len(samples))
        if processed % 20 == 0 or processed == len(samples):
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            log.info(
                "Progress: %d/%d samples (%.1f samples/s, %d questions, %d failed)",
                processed, len(samples), rate, total_questions, failed,
            )

    elapsed = time.time() - t0
    log.info(
        "Question generation complete: %d samples, %d questions, %d failed (%.1fs)",
        len(samples), total_questions, failed, elapsed,
    )
    return samples


# ─── JSONL EXPORT ─────────────────────────────────────────────────────────────

def _flatten_to_records(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert chunked samples (with generated_qas) OR pre-flattened records to JSONL rows."""
    records = []
    for sample in samples:
        qas = sample.get("generated_qas")
        if qas is not None:
            # Generated via generate_questions_for_samples
            for qa in qas:
                records.append({
                    "question": qa.get("question", ""),
                    "reference_answer": qa.get("answer", ""),
                    "reference_context": [sample["text"]],
                    "reference_cids": [sample["cid"]],
                    "intent": qa.get("intent", "UNKNOWN"),
                    "is_negative": qa.get("intent") == "NEGATIVE",
                    "legal_type": sample.get("legal_type", ""),
                    "doc_id": sample.get("doc_id"),
                })
        elif "question" in sample:
            # Pre-flattened record (from negative gen or validate_samples)
            records.append({
                "question": sample.get("question", ""),
                "reference_answer": sample.get("reference_answer", ""),
                "reference_context": [sample["text"]],
                "reference_cids": [sample["cid"]],
                "intent": sample.get("intent", "UNKNOWN"),
                "is_negative": sample.get("is_negative", False),
                "legal_type": sample.get("legal_type", ""),
                "doc_id": sample.get("doc_id"),
                "tier": sample.get("tier"),
                "critic": sample.get("critic"),
            })
        else:
            # Stub (no generation)
            records.append({
                "question": "",
                "reference_answer": "",
                "reference_context": [sample["text"]],
                "reference_cids": [sample["cid"]],
                "intent": "",
                "is_negative": False,
                "legal_type": sample.get("legal_type", ""),
                "doc_id": sample.get("doc_id"),
            })
    return records


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> int:
    """Write records as JSONL in the golden set schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_review_csv(flagged: List[Dict[str, Any]], output_path: Path) -> int:
    """Export flagged records as CSV for human (SME) review.

    Columns: question, reference_answer, source_chunk, intent, legal_type,
             critic_answerability, critic_correctness, critic_clarity,
             critic_reasoning, sme_verdict (empty — filled by reviewer).
    """
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not flagged:
        return 0

    fieldnames = [
        "question", "reference_answer", "source_chunk", "intent", "legal_type",
        "critic_answerability", "critic_correctness", "critic_clarity",
        "critic_reasoning", "sme_verdict",
    ]
    count = 0
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in flagged:
            critic = rec.get("critic") or {}
            writer.writerow({
                "question": rec.get("question", ""),
                "reference_answer": rec.get("reference_answer", ""),
                "source_chunk": rec.get("text", "")[:500],
                "intent": rec.get("intent", ""),
                "legal_type": rec.get("legal_type", ""),
                "critic_answerability": critic.get("answerability", ""),
                "critic_correctness": critic.get("correctness", ""),
                "critic_clarity": critic.get("clarity", ""),
                "critic_reasoning": critic.get("reasoning", critic.get("error", "")),
                "sme_verdict": "",  # filled by reviewer: approve / edit / reject
            })
            count += 1
    return count


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a validated silver/golden RAGAS evaluation dataset from Qdrant",
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
        "--target", type=int, default=DEFAULT_TARGET,
        help=f"Target number of chunks to sample (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--min-text-len", type=int, default=DEFAULT_MIN_TEXT_LEN,
        help=f"Minimum chunk text length in characters (default: {DEFAULT_MIN_TEXT_LEN})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducible sampling (default: {DEFAULT_SEED})",
    )

    llm_group = parser.add_argument_group("LLM question generation (optional)")
    llm_group.add_argument(
        "--generate-questions", action="store_true",
        help="Generate questions and answers via MiMo v2.5 (requires --api-key)",
    )
    llm_group.add_argument(
        "--api-key", default=None,
        help="MiMo / OpenAI-compatible API key (or set MIMO_API_KEY / OPENAI_API_KEY env var)",
    )
    llm_group.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=f"LLM API base URL (default: {DEFAULT_BASE_URL})",
    )
    llm_group.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Generator model name (default: {DEFAULT_MODEL})",
    )
    llm_group.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Chunks per LLM batch (default: {DEFAULT_BATCH_SIZE})",
    )
    llm_group.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM calls (default: {DEFAULT_CONCURRENCY})",
    )
    llm_group.add_argument(
        "--neg-ratio", type=float, default=DEFAULT_NEG_RATIO,
        help=f"Fraction of target to generate as negative (unanswerable) samples (default: {DEFAULT_NEG_RATIO})",
    )

    critic_group = parser.add_argument_group("Critic validation (MiMo v2.5 Pro)")
    critic_group.add_argument(
        "--validate", action="store_true",
        help="Run MiMo v2.5 Pro critic validation on generated Q&A pairs",
    )
    critic_group.add_argument(
        "--critic-model", default=DEFAULT_CRITIC_MODEL,
        help=f"Critic model name (default: {DEFAULT_CRITIC_MODEL})",
    )

    parser.add_argument(
        "--output", default=None,
        help="Output JSONL file path (default: eval_results/silver_dataset_{timestamp}.jsonl)",
    )
    return parser


def main() -> int:
    import os
    parser = build_parser()
    args = parser.parse_args()

    # Connect to Qdrant
    try:
        client = QdrantClient(url=args.qdrant_url, timeout=120.0)
        client.get_collections()
        log.info("Connected to Qdrant at %s ✓", args.qdrant_url)
    except Exception as exc:
        log.error("Cannot connect to Qdrant at %s: %s", args.qdrant_url, exc)
        return 1

    # Step 1: Sample chunks
    samples = sample_chunks(
        client=client,
        collection=args.collection,
        target=args.target,
        min_text_len=args.min_text_len,
        seed=args.seed,
    )
    if not samples:
        log.error("No samples collected. Exiting.")
        return 1

    # Step 2: Generate questions + negative samples
    all_records: List[Dict[str, Any]] = []

    if args.generate_questions:
        api_key = args.api_key or os.environ.get("MIMO_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.error("--api-key is required (or set MIMO_API_KEY env var)")
            return 1

        log.info("Generator: %s @ %s (concurrency=%d)", args.model, args.base_url, args.concurrency)

        # 2a: Generate regular Q&A
        samples = asyncio.run(generate_questions_for_samples(
            samples=samples,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        ))
        all_records = _flatten_to_records(samples)

        # 2b: Generate negative samples (~neg_ratio of target)
        neg_count = max(1, int(args.target * args.neg_ratio))
        log.info("Generating %d negative (unanswerable) samples...", neg_count)
        neg_records = asyncio.run(generate_negative_samples(
            samples=samples,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            neg_count=neg_count,
            concurrency=args.concurrency,
        ))
        all_records.extend(neg_records)
        log.info("Total Q&A records before validation: %d", len(all_records))

        # Step 3: Critic validation (MiMo v2.5 Pro)
        if args.validate:
            log.info("Critic: %s (validating %d records)...", args.critic_model, len(all_records))
            kept, flagged, rejected = asyncio.run(validate_samples(
                qa_records=all_records,
                api_key=api_key,
                base_url=args.base_url,
                critic_model=args.critic_model,
                concurrency=args.concurrency,
            ))

            # Export human-review CSV for flagged samples
            if flagged:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                review_path = OUTPUT_DIR / f"review_flagged_{timestamp}.csv"
                n_review = write_review_csv(flagged, review_path)
                log.info("Human review CSV: %d flagged samples → %s", n_review, review_path)

            # Export rejected log
            if rejected:
                rej_path = OUTPUT_DIR / f"rejected_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
                write_jsonl(rejected, rej_path)
                log.info("Rejected log: %d records → %s", len(rejected), rej_path)

            # Only kept samples go to main output
            all_records = kept
        else:
            log.info("Skipping critic validation (--validate not set). Output = raw silver.")
    else:
        log.info("Skipping question generation. Output = stub records for manual review.")
        all_records = _flatten_to_records(samples)

    # Step 4: Export JSONL
    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tier = "validated_silver" if (args.generate_questions and args.validate) else "silver" if args.generate_questions else "stub"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"{tier}_dataset_{timestamp}.jsonl"

    count = write_jsonl(all_records, output_path)

    # Summary
    intent_dist = dict(Counter(r.get("intent", "UNKNOWN") for r in all_records))
    neg_total = sum(1 for r in all_records if r.get("is_negative"))

    print()
    print("=" * 60)
    tier_label = "VALIDATED SILVER" if (args.generate_questions and args.validate) else "RAW SILVER" if args.generate_questions else "STUB"
    print(f"✅ DATASET BUILT [{tier_label}]")
    print(f"   Chunks sampled      : {len(samples)}")
    print(f"   Output records      : {count}")
    print(f"   Negative samples    : {neg_total}")
    print(f"   Intent distribution : {intent_dist}")
    print(f"   Output file         : {output_path}")
    print("=" * 60)

    return 0



if __name__ == "__main__":
    sys.exit(main())

