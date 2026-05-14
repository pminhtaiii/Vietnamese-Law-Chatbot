# Law Chatbot — RAG Evaluation

A Vietnamese legal chatbot that retrieves relevant law articles from a Qdrant vector database and generates answers using an LLM. This context covers the RAG evaluation domain — how retrieval quality is measured and improved.

## Language

**Corpus**:
The live Qdrant collection (`legal_docs`) that the backend retriever queries at runtime.
_Avoid_: database, store, index (too generic)

**Chunk**:
A unit of indexed legal text stored in Qdrant, identified by an integer `cid` (Qdrant point ID). Contains `text`, metadata (e.g., `legal_type`, `doc_id`), and dense+sparse vectors.
_Avoid_: document, record, passage

**Golden set**:
An evaluation dataset built from the corpus. Each sample contains a question, reference answer, reference context, and reference CIDs. Used as ground truth for RAGAS scoring.
_Avoid_: test set, benchmark, train.csv (legacy)

**Legal type**:
Classification of Vietnamese legal documents (Luật, Nghị định, Thông tư, Quyết định, etc.). Used for stratified sampling of the golden set and as payload metadata in Qdrant.
_Avoid_: document type, category

**Intent**:
The user's question purpose, classified by QueryReflector into 7 types: LEGAL_LOOKUP, COMPARE, SUMMARIZE, PROCEDURE, DEFINITION, MULTI_HOP, CHITCHAT. Each intent triggers a different retrieval strategy.
_Avoid_: query type, category (too vague)

**Hybrid search**:
The retrieval method: BGE-M3 dense (1024-dim) + sparse vectors fused via Reciprocal Rank Fusion in Qdrant.
_Avoid_: vector search (misses the sparse arm), semantic search

**Rerank**:
Post-retrieval scoring step using either a local Vietnamese reranker (`thanhtantran/Vietnamese_Reranker`) or Cohere `rerank-v3.5` as fallback. Scores determine whether web search fallback is triggered.
_Avoid_: re-scoring, re-ranking (inconsistent casing)

## Relationships

- A **Chunk** belongs to one **Legal type** and one document (`doc_id`)
- A **Golden set** sample is derived from one or more **Chunks** (via `reference_cids`)
- An **Intent** determines which retrieval strategy the **Hybrid search** + **Rerank** pipeline uses
- A **Corpus** contains all **Chunks**; the **Golden set** is a small sample drawn from it

## Example dialogue

> **Dev:** "When building the **Golden set**, do we sample from `vietnamese_laws_m3` or `legal_docs`?"
> **Domain expert:** "`legal_docs` — that's what the retriever queries in production. `vietnamese_laws_m3` is the pipeline staging collection."

> **Dev:** "Should we import the **Retriever** directly or call the API for evaluation?"
> **Domain expert:** "API. We want production fidelity — intent routing, **Rerank**, web fallback, the whole path."

## Flagged ambiguities

- `vietnamese_laws_m3` (pipeline collection) vs `legal_docs` (retriever collection) — resolved: evaluation uses `legal_docs` only, as it's the production retrieval source.
- `train.csv` was previously used as the golden set — resolved: it's stale and replaced by corpus-derived golden set described in `docs/rag_dataset_vector_db_guide.md`.