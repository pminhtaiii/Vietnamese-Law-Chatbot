\"\"\"
validate_golden_dataset.py — Standalone critic validation for golden datasets
=============================================================================

Load an existing JSONL golden dataset, run the MiMo v2.5 Pro critic validation
on each sample, and output a filtered dataset plus a human-review CSV.

Usage:
  python evaluation/validate_golden_dataset.py --input my_dataset.jsonl --api-key KEY
\"\"\"

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Re-use constants and logic from build_golden_dataset if possible, 
# but keeping it standalone for simplicity in execution.

DEFAULT_BASE_URL = \"https://api.xiaomimimo.com/v1\"
DEFAULT_CRITIC_MODEL = \"mimo-v2.5-pro\"
DEFAULT_CONCURRENCY = 5

logging.basicConfig(
    level=logging.INFO,
    format=\"%(asctime)s | %(levelname)-7s | %(message)s\",
    datefmt=\"%H:%M:%S\",
)
log = logging.getLogger(\"golden_validator\")

CRITIC_SYSTEM_PROMPT = \"\"\"You are a strict quality validator for a Vietnamese legal RAG evaluation dataset.

Given a source chunk and a generated Q&A pair, score each dimension 1-5:
- answerability (1-5): Can the question be fully answered from THIS chunk alone? (5=yes completely, 1=not at all)
- correctness (1-5): Is the reference_answer factually accurate per the chunk? (5=perfectly correct, 1=wrong/hallucinated)
- clarity (1-5): Is the question clear, natural Vietnamese, unambiguous? (5=excellent, 1=confusing)

For NEGATIVE intent questions: answerability should be LOW (1-2) since they are intentionally unanswerable.

Return ONLY compact JSON, no markdown:
{\"answerability\": N, \"correctness\": N, \"clarity\": N, \"reasoning\": \"<1 sentence>\"}\"\"\"

def make_critic_prompt(text: str, question: str, answer: str, intent: str) -> str:
    return (
        f\"SOURCE CHUNK:\\n{text}\\n\\n\"
        f\"QUESTION (intent={intent}): {question}\\n\"
        f\"REFERENCE ANSWER: {answer}\\n\\n\"
        f\"Validate this Q&A pair against the source chunk.\"
    )

async def _call_critic(client, model, prompt, semaphore):
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {\"role\": \"system\", \"content\": CRITIC_SYSTEM_PROMPT},
                        {\"role\": \"user\", \"content\": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
            content = response.choices[0].message.content.strip()
            if content.startswith(\"```\"):
                lines = content.split(\"\\n\")
                content = \"\\n\".join(lines[1:-1] if lines[-1].strip() == \"```\" else lines[1:])
            return json.loads(content)
        except Exception as exc:
            log.warning(f\"Critic attempt {attempt+1} failed: {exc}\")
            await asyncio.sleep(1)
    return None

async def validate_dataset(
    input_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    concurrency: int
):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(concurrency)

    records = []
    with open(input_path, encoding=\"utf-8\") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    log.info(f\"Loaded {len(records)} records from {input_path}\")
    
    tasks = []
    for rec in records:
        # reference_context is usually a list
        text = \"\\n\\n\".join(rec[\"reference_context\"]) if isinstance(rec[\"reference_context\"], list) else str(rec[\"reference_context\"])
        prompt = make_critic_prompt(text, rec[\"question\"], rec[\"reference_answer\"], rec.get(\"intent\", \"UNKNOWN\"))
        tasks.append(_call_critic(client, model, prompt, semaphore))

    results = await asyncio.gather(*tasks)

    kept, flagged, rejected = [], [], []
    for rec, scores in zip(records, results):
        if scores is None:
            rec[\"critic\"] = {\"error\": \"failed\"}
            rec[\"tier\"] = \"flagged\"
            flagged.append(rec)
            continue
        
        rec[\"critic\"] = scores
        a = int(scores.get(\"answerability\", 0))
        c = int(scores.get(\"correctness\", 0))
        cl = int(scores.get(\"clarity\", 0))
        intent = rec.get(\"intent\", \"\")

        if intent == \"NEGATIVE\":
            if a <= 2 and cl >= 3:
                rec[\"tier\"] = \"kept\"; kept.append(rec)
            elif cl < 2:
                rec[\"tier\"] = \"rejected\"; rejected.append(rec)
            else:
                rec[\"tier\"] = \"flagged\"; flagged.append(rec)
        elif min(a, c, cl) == 1:
            rec[\"tier\"] = \"rejected\"; rejected.append(rec)
        elif min(a, c, cl) == 2:
            rec[\"tier\"] = \"flagged\"; flagged.append(rec)
        else:
            rec[\"tier\"] = \"kept\"; kept.append(rec)

    return kept, flagged, rejected

def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, \"w\", encoding=\"utf-8\") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + \"\\n\")

def write_review_csv(flagged, path):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not flagged: return
    fieldnames = [\"question\", \"reference_answer\", \"intent\", \"critic_answerability\", \"critic_correctness\", \"critic_clarity\", \"critic_reasoning\", \"sme_verdict\"]
    with open(path, \"w\", encoding=\"utf-8-sig\", newline=\"\") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in flagged:
            c = r.get(\"critic\", {})
            writer.writerow({
                \"question\": r.get(\"question\", \"\"),
                \"reference_answer\": r.get(\"reference_answer\", \"\"),
                \"intent\": r.get(\"intent\", \"\"),
                \"critic_answerability\": c.get(\"answerability\", \"\"),
                \"critic_correctness\": c.get(\"correctness\", \"\"),
                \"critic_clarity\": c.get(\"clarity\", \"\"),
                \"critic_reasoning\": c.get(\"reasoning\", \"\"),
                \"sme_verdict\": \"\"
            })

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--input\", required=True, help=\"Input JSONL file\")
    parser.add_argument(\"--api-key\", help=\"MiMo API Key\")
    parser.add_argument(\"--base-url\", default=DEFAULT_BASE_URL)
    parser.add_argument(\"--model\", default=DEFAULT_CRITIC_MODEL)
    parser.add_argument(\"--concurrency\", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(\"MIMO_API_KEY\") or os.environ.get(\"OPENAI_API_KEY\")
    if not api_key:
        print(\"Error: --api-key required\")
        return

    input_path = Path(args.input)
    kept, flagged, rejected = await validate_dataset(input_path, api_key, args.base_url, args.model, args.concurrency)

    stem = input_path.stem
    out_dir = input_path.parent / \"validated\"
    out_dir.mkdir(exist_ok=True)

    write_jsonl(kept, out_dir / f\"{stem}_kept.jsonl\")
    write_jsonl(rejected, out_dir / f\"{stem}_rejected.jsonl\")
    write_review_csv(flagged, out_dir / f\"{stem}_flagged_review.csv\")

    print(f\"Validation complete:\")
    print(f\"  Kept:     {len(kept)}\")
    print(f\"  Flagged:  {len(flagged)} (see CSV)\")
    print(f\"  Rejected: {len(rejected)}\")
    print(f\"Results saved to {out_dir}\")

if __name__ == \"__main__\":
    asyncio.run(main())
