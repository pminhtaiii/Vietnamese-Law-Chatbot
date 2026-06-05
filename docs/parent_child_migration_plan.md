# Parent-Child Retrieval Architecture Migration Plan

## 1. Goal & Context
Our current VectorRAG pipeline uses large 2400-token chunks. While this provides excellent macro-context for the LLM, it suffers from "embedding dilution" (1024-dim BGE-M3 vectors struggle to accurately represent 2400 tokens of dense legal text). We are migrating to a **Parent-Child Retriever** architecture to achieve micro-precision in vector search while preserving macro-context for the LLM.

## 2. Data Pipeline (Kaggle/SQLite) Changes
- **Two-Pass Chunking:**
  1. Chunk documents into **Parents** (approx. 2400 tokens or by logical legal boundaries). Assign a unique `parent_id`.
  2. Sub-chunk Parents into **Children** (approx. 400 tokens, 50-token overlap). Assign a `cid` (Child ID).
- **Embedding Generation:** Only the 400-token Child chunks will be passed through BGE-M3 to generate dense and sparse vectors.
- **SQLite Schema Update:** The `chunks_XXXX.sqlite` output will store Child points. It must include `parent_id` and the FULL `parent_text` alongside the Child's vectors and `child_text`.

## 3. Vector Database (Qdrant) Strategy
- **"Self-Contained" Qdrant Points:** We will NOT use a secondary Key-Value database to store Parent documents.
- **Payload Duplication:** Every Child point in Qdrant will store the full 2400-token `parent_text` in its payload.
- **Resource Impact:** Because our Qdrant uses `ON_DISK_PAYLOAD=true` and INT8 Scalar Quantization, duplicating the parent text across 5-6 children will slightly increase Disk Usage but will have **zero impact on RAM**.

## 4. Backend Retriever Changes (`retriever.py`)
- **Query Phase:** `_hybrid_search` will query Qdrant using the BGE-M3 embeddings and retrieve the top-K *Children* (e.g., fetch top 15-20).
- **Deduplication Phase:** Group the retrieved Children by `parent_id`. Keep only the highest-scoring Child per Parent to avoid passing redundant context to the Reranker/LLM.
- **Reranking Phase:** Extract the `parent_text` from the deduplicated points. Pass these unique Parents to the Reranker (Cohere/Local). The LLM ultimately receives the top-5 Parents.

## 5. Areas of Uncertainty (Focus for Grilling)
- **Reranker Input:** Should the local Vietnamese Reranker score the query against the 400-token `child_text` (for pure precision) or the 2400-token `parent_text` (to catch holistic relevance)?
- **Chunk Boundaries:** How do we handle Vietnamese legal articles that naturally split awkwardly across a 2400-token Parent boundary? 
- **GraphRAG Alignment:** How does changing the `cid` from representing a 2400-token chunk to a 400-token chunk affect our existing GraphRAG pipeline and community summaries?
