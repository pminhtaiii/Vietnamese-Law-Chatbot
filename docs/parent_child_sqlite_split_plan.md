# Implementation Plan: Parent-Child SQLite Split + Lean Qdrant

## Problem Statement

The current `notebook_parent_child_pipeline.py` stores **every child row** with a full copy of `parent_text` (~8000 chars) in two places:
1. **SQLite** — each of the ~50K rows per shard carries `parent_text`, causing massive file bloat.
2. **Qdrant** — every point's payload carries `parent_text`, exceeding the 5GB Docker memory limit.

With ~500K children and ~80K parents (avg 6.25 children/parent), `parent_text` is duplicated ~6x per parent, consuming ~4GB of unnecessary storage in SQLite and making Qdrant unpushable.

## Solution Architecture: Two-File SQLite + Lean Qdrant

### Storage Split

**`parents.sqlite`** — one file, one row per parent chunk:
```
parent_id       INTEGER PRIMARY KEY
doc_id          INTEGER
parent_text     TEXT        -- full article text (~8000 chars), stored ONCE
document_number TEXT
title           TEXT
legal_type      TEXT
legal_sectors   TEXT
issuing_authority TEXT
issuance_date   TEXT
url             TEXT
```

**`children_XXXX.sqlite`** — sharded files, one row per child chunk:
```
cid             INTEGER PRIMARY KEY
parent_id       INTEGER     -- FK to parents.sqlite
doc_id          INTEGER
parent_index    INTEGER
child_index     INTEGER
text            TEXT        -- child text only (~1400 chars)
dense_vector    BLOB
sparse_indices  TEXT
sparse_values   TEXT
```

**Qdrant collection** — lean payload, dense vectors:
```
point ID:       cid (child chunk ID)
vector:         dense_vector (1024-dim float32)
payload:
    parent_id:  int         -- FK to parents.sqlite (8 bytes, not 8KB)
    text:       str         -- child text for reranker (~1400 chars)
    doc_id:     int
    title:      str         -- for filtering
    legal_type: str         -- for filtering
```

### Qdrant Payload Size Comparison

| Field              | Before     | After       |
|--------------------|------------|-------------|
| dense_vector       | 4 KB       | 4 KB        |
| sparse_vector      | ~2 KB      | ~2 KB       |
| parent_text        | ~8 KB      | **0**       |
| text (child)       | ~1.4 KB    | ~1.4 KB     |
| metadata           | ~0.5 KB    | ~0.3 KB     |
| **Total per point**| **~16 KB** | **~8 KB**   |

50% reduction per point → Qdrant fits in 5GB Docker limit.

### Query-Time Flow

```
User query
  → BGE-M3 encode query (dense + sparse)
  → Qdrant hybrid search (top-K children)
  → Reranker scores on child `text`
  → Top-N children selected
  → Batch-lookup parent_id → parent_text from parents.sqlite (via ParentStore)
  → LLM receives parent_text as context with [ID: <cid>] citations
```

## Files to Modify

### 1. `data_pipelines/vector_rag/embeddings/notebook_parent_child_pipeline.py`

**Changes to `SqliteWriter`:**
- Split into two SQLite connections: `_parents_conn` (parents.sqlite) and `_children_conn` (children_XXXX.sqlite).
- `_parents_conn` writes a `parents` table with deduplicated parent records.
- `_children_conn` writes a `children` table with child text + vectors only (no `parent_text`).
- `insert_points()` logic: write parent row once (INSERT OR IGNORE), then write child row.

**No changes** to chunking logic, embedding logic, checkpoint logic, or pipeline flow. Only the writer changes.

### 2. `data_pipelines/vector_rag/push_sqlite_to_qdrant.py`

- Glob `children_*.sqlite` instead of `chunks_*.sqlite`.
- Payload: include `parent_id`, `text`, `doc_id`, `title`, `legal_type`. Remove `parent_text`.
- Keep sparse vector support as-is.

### 3. `backend/app/services/parent_store.py` (NEW FILE)

```python
class ParentStore:
    """SQLite-backed parent text lookup. Loads parent_id → parent_text dict at startup."""
    def __init__(self, db_path: str): ...
    def get_parent_texts(self, parent_ids: list[int]) -> dict[int, str]: ...
    def close(self): ...
```

For ~80K parents with avg 8KB text, in-memory dict is ~640MB. If that's too large, fall back to on-demand SQLite lookups with LRU cache. Start with in-memory; optimize if needed.

### 4. `backend/app/services/retriever.py`

- Add `parent_store: Optional[ParentStore]` to `__init__`.
- After Qdrant retrieval, extract `parent_id` from each result's payload.
- After reranking and selecting top-N, batch-lookup `parent_text` via `parent_store.get_parent_texts()`.
- Set `doc["text"] = parent_text` and `doc["child_text"] = child_text`.
- Generator.py receives `parent_text` through the existing `_build_context()` — no generator changes needed.

### 5. `backend/app/core/config.py`

Add:
```python
PARENTS_DB_PATH: str = os.getenv("PARENTS_DB_PATH", str(BASE_DIR / "data" / "parents.sqlite"))
```

### 6. `backend/app/main.py`

- Initialize `ParentStore(settings.PARENTS_DB_PATH)` at startup.
- Pass to `Retriever(..., parent_store=parent_store)`.

### 7. `docker-compose.yml`

Mount parents.sqlite into backend container:
```yaml
backend-api:
  volumes:
    - ./data/parents.sqlite:/app/data/parents.sqlite:ro
```

## Implementation Order

1. **Split SQLite writer** — `notebook_parent_child_pipeline.py`. Foundation for everything else.
   - Verify: Run with `max_content_docs=100`, check `parents.sqlite` + `children_0000.sqlite` exist with correct schemas.

2. **Update Qdrant pusher** — `push_sqlite_to_qdrant.py`. Reads new format, pushes lean payload.
   - Verify: Qdrant points have `parent_id`, no `parent_text`.

3. **Create `ParentStore`** — `backend/app/services/parent_store.py`.
   - Verify: Unit test with small parents.sqlite.

4. **Update Retriever** — `backend/app/services/retriever.py`. Inject ParentStore, add post-rerank parent lookup.
   - Verify: Query returns docs with full parent_text.

5. **Wire config + startup** — `config.py`, `main.py`.

6. **Update Docker** — `docker-compose.yml`. Mount parents.sqlite.

7. **End-to-end validation** — Full pipeline → push → query → verify LLM context.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| ParentStore RAM usage (~640MB for 80K parents) | Fall back to SQLite + LRU cache if RAM is tight in Docker |
| parents.sqlite not mounted in Docker | Explicit volume mount in docker-compose.yml |
| Checkpoint compatibility with old format | New checkpoint file path or reset checkpoint on first run with new format |
| Sparse vector size in Qdrant | Consider dense-only retrieval if sparse still exceeds limits |