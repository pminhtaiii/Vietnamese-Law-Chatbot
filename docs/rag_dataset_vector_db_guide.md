# RAGAS Evaluation Path for an Indexed Corpus

## Goal

Use the corpus already stored in Qdrant (`legal_docs` collection) to build a golden dataset, then evaluate RAG quality end-to-end via the live API using RAGAS.

## Core rules

1. **Source of truth:** The `legal_docs` collection in Qdrant is the retrieval source. This is what the backend retriever (`backend/app/services/retriever.py`) queries at runtime.
2. **Replaces legacy evaluation:** This path replaces the old `train.csv`-based evaluation in `evaluation/ragas_evaluator.py`. The legacy `data/raw_csv/train.csv` golden set is stale and no longer represents the live corpus.
3. **End-to-end evaluation:** Run evaluation through the live API endpoint (`POST /api/retrieve`) — not by importing retriever code directly. This measures production fidelity (intent routing, hybrid search, reranking, web fallback).
4. **Golden set is built FROM the corpus, not outside it.** Do not use legacy QA files as the benchmark source.

## Golden set schema

Every golden sample must include all of these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | yes | The evaluation question, phrased as a user would ask it |
| `reference_answer` | string | yes | Expected answer grounded in the source chunk(s) |
| `reference_context` | list[string] | yes | Chunk text(s) the question was derived from |
| `reference_cids` | list[int] | yes | Qdrant point IDs of the source chunks |

Export format: JSONL (one record per line) or CSV.

**`reference_cids` definition:** These are the chunk IDs from which the question was derived — verifiable by construction. They serve as the expected-retrieval target for RAGAS context recall scoring.

**`reference_answer` is required**, not optional. RAGAS context_recall cannot be computed without a reference answer.

## Recommended path

### Step 1 — Sample representative chunks from Qdrant

Scroll the `legal_docs` collection with **stratified sampling by `legal_type`**:

1. Query distinct `legal_type` values in the collection (e.g., Luật, Nghị định, Thông tư, Quyết định, …).
2. For each type, scroll a proportional share of chunks (or a minimum floor of 3 per type).
3. Target: **50-100 chunks** total for the full golden set.
4. Deduplicate by `doc_id` — skip chunks from a document already sampled.
5. Filter out chunks with < 100 characters of `text` content (too short to generate meaningful questions from).

**Concrete approach:**
```
for each legal_type:
    scroll(payload_filter=FieldCondition(key="legal_type", match=MatchValue(value=legal_type)), limit=N)
    deduplicate by doc_id
    skip if len(text) < 100
```

### Step 2 — Write 1-3 answerable questions per chunk

For each sampled chunk:
- Write questions that are **directly answerable from the chunk text**.
- Cover real user intents: definition, procedure, condition, comparison.
- Avoid ambiguity and questions requiring outside knowledge.
- Prefer Vietnamese phrasing that matches real user queries (check logs for common patterns).

### Step 3 — Fill the golden set record

For each question, populate:
- `reference_answer`: a concise answer that could be extracted from the chunk text.
- `reference_context`: the chunk text(s) the question was derived from.
- `reference_cids`: the Qdrant point ID(s) of those chunks.

### Step 4 — Export as JSONL

One record per line:
```json
{"question": "Điều kiện để được miễn thuế thu nhập cá nhân là gì?", "reference_answer": "Cá nhân được miễn thuế khi...", "reference_context": ["Điều 4. Thu nhập được miễn thuế..."], "reference_cids": [12345]}
```

### Step 5 — Run the live API evaluator

The evaluator should:
1. Read golden questions from the JSONL file.
2. For each question, call `POST /api/retrieve` with the question string.
3. Collect returned `sources[].text` as retrieved contexts.
4. Collect the generated answer (from `POST /api/chat` if measuring generation, or from `POST /api/retrieve` if measuring retrieval only).
5. Build RAGAS evaluation data:
   - `question` → from golden set
   - `answer` → from API response
   - `contexts` → list of retrieved chunk texts
   - `reference` → from golden set (`reference_answer`)
6. Run RAGAS metrics.

### Step 6 — Compute and review RAGAS metrics

| Metric | What it measures | Failure signal |
|--------|-----------------|----------------|
| **Context Recall** | Did the retriever find the relevant chunks? | Retriever missing the source chunks |
| **Faithfulness** | Does the answer use only retrieved context? | Generator hallucinating |
| **Answer Relevancy** | Does the answer address the question? | Answer is off-topic |

**Always review the worst cases**, not only the average score. Sort by per-question score and examine the bottom 10%.

## What makes a good golden sample

- Answerable directly from the indexed `legal_docs` corpus.
- Grounded in one clear chunk or a small cluster of related chunks.
- Covers real user intents: definition, procedure, condition, comparison, multi-hop.
- Avoids ambiguity and outside knowledge.
- Has a `reference_answer` that a human reviewer would agree is correct given the `reference_context`.

## Practical fallback

If you cannot build the full 50-100 sample golden set yet, create a **temporary subset of 20-50 manually reviewed samples** and evaluate that first. Even 20 high-quality samples will surface major retrieval or generation issues.

## Known constraints

- The `legal_docs` collection uses named vectors (`dense` 1024-dim + `sparse`) with INT8 scalar quantization. Sampling via scroll does not require encoding queries — just payload filters.
- The retriever applies intent routing (7 intent types) via `QueryReflector` before searching. Different intents trigger different retrieval strategies (e.g., COMPARE runs parallel searches, MULTI_HOP chains searches). Golden questions should ideally cover at least LEGAL_LOOKUP, PROCEDURE, and COMPARE intents to exercise different paths.
- Web fallback (Tavily) may be triggered if rerank scores fall below `RAG_FALLBACK_SCORE_THRESHOLD`. For evaluation of local corpus retrieval, you may want to disable web fallback or filter out `used_web=true` responses.

## Internal sources used

- `data_pipelines/vector_rag/push_sqlite_to_qdrant.py` — ingestion pipeline, `legal_docs` collection schema
- `data_pipelines/vector_rag/migrate.py` — local-to-Docker Qdrant migration
- `backend/app/services/retriever.py` — hybrid search (dense+sparse RRF) + rerank (Vietnamese local / Cohere)
- `backend/app/services/query_router.py` — intent-based retrieval routing
- `backend/app/main.py` — `POST /api/retrieve` endpoint (retrieval-only)
- `evaluation/ragas_evaluator.py` — legacy evaluator (replaced by this guide)
- `evaluation/legal_rag_evaluator.py` — custom legal metrics
- `docs/implementation_plan.md.resolved` — original targets (faithfulness ≥ 0.80, relevancy ≥ 0.75, context precision ≥ 0.70, context recall ≥ 0.70)