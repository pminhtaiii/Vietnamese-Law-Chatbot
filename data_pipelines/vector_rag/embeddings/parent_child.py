# -*- coding: utf-8 -*-
"""notebook_parent_child_pipeline.py

Parent-Child Retrieval Pipeline for Vietnamese Legal Documents.

Architecture (Small-to-Big Retrieval, Liu et al. 2023):
- Parent Chunks: logically complete Articles (Dieu) batched up to ~8000 chars (~2400 Vietnamese tokens).
- Child Chunks: 1400-char segments (400 Vietnamese tokens) with 260-char overlap (75 tokens).
- Qdrant points store Child vectors with lean payload (cid, parent_id, text, doc_id).
- parent_text lives in a separate parents.sqlite — looked up by ParentStore at query time.
- cid in payload = parent_id, preserving citation consistency in generator.py.
"""

# !pip install -qU "transformers<4.45.0" "huggingface_hub>=0.23.2,<1.0" qdrant-client
# !pip install -qU datasets pyarrow pyyaml tqdm pandas

import os
import gc
import re
import json
import time
import html
import hashlib
import sqlite3
import struct
import logging
from pathlib import Path
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Iterable, Iterator, Optional

import yaml
import torch

from transformers import AutoModel
if not getattr(AutoModel, '_dtype_patch_applied', False):
    _orig_from_pretrained = AutoModel.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.pop('dtype', None)
        kwargs.pop('torch_dtype', None)
        return _orig_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    AutoModel.from_pretrained = _patched_from_pretrained
    AutoModel._dtype_patch_applied = True

import pandas as pd
from tqdm import tqdm
from datasets import DownloadConfig, load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from huggingface_hub import snapshot_download

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('parent_child_pipeline')

@dataclass
class PipelineConfig:
    hf_dataset_id: str = 'th1nhng0/vietnamese-legal-documents'
    hf_config_content: str = 'content'
    hf_config_metadata: str = 'metadata'
    hf_split: str = 'legacy'
    hf_token: Optional[str] = None
    hf_content_field: str = 'content_html'

    download_cache_dir: str = '/kaggle/working/hf_cache'
    max_download_retries: int = 3
    max_content_docs: int = 0

    min_content_chars: int = 300
    max_content_chars: int = 500_000
    min_meaningful_lines: int = 3
    skip_legal_types: list[str] = field(default_factory=lambda: [
        'Cong van', 'Thong bao', 'To trinh',
        'Công văn', 'Thông báo', 'Tờ trình',
    ])
    skip_before_year: int = 1986

    # Parent-Child chunking
    # Vietnamese text: ~3.5 chars/token → 8000 chars ≈ 2400 tokens (parent), 1400 chars ≈ 400 tokens (child)
    parent_size_chars: int = 8000
    child_size_chars: int = 1400
    child_overlap_chars: int = 260
    min_child_chars: int = 200

    embedding_model: str = 'huyydangg/DEk21_hcmute_embedding'
    embedding_dim: int = 768
    embedding_batch_size: int = 64
    embedding_max_length: int = 256
    use_fp16: bool = torch.cuda.is_available()

    sqlite_output_dir: str = '/kaggle/working/sqlite_outputs_parent_child'
    sqlite_rows_per_db: int = 50000
    dense_vector_name: str = 'dense'
    sparse_vector_name: str = 'sparse'

    checkpoint_file: str = '/kaggle/working/parent_child_checkpoint.json'
    checkpoint_every_docs: int = 200
    flush_chunks_threshold: int = 512

    dry_run: bool = False
    resume: bool = False
    log_every: int = 100


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    cfg = PipelineConfig()
    if not config_path:
        return cfg
    p = Path(config_path)
    if not p.exists():
        return cfg
    with open(p, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


cfg = load_config(None)
print(cfg)


class Checkpoint:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data: dict[str, Any] = {
            'processed_doc_ids': [],
            'total_chunks_pushed': 0,
            'started_at': time.time(),
            'last_updated_at': time.time(),
        }
        self._processed_set: set[int] = set()

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with open(self.path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        raw_ids = self.data.get('processed_doc_ids', [])
        self._processed_set = {int(x) for x in raw_ids}
        self.data['processed_doc_ids'] = sorted(self._processed_set)
        log.info('Checkpoint loaded: docs_done=%d chunks_pushed=%d', len(self._processed_set), int(self.data.get('total_chunks_pushed', 0)))
        return True

    def save(self) -> None:
        self.data['processed_doc_ids'] = sorted(self._processed_set)
        self.data['last_updated_at'] = time.time()
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False)
        tmp.replace(self.path)

    def is_done(self, doc_id: int) -> bool:
        return doc_id in self._processed_set

    def mark_done(self, doc_ids: Iterable[int], chunks_added: int = 0) -> None:
        for doc_id in doc_ids:
            self._processed_set.add(int(doc_id))
        if chunks_added:
            self.data['total_chunks_pushed'] = int(self.data.get('total_chunks_pushed', 0)) + int(chunks_added)

    @property
    def total_chunks_pushed(self) -> int:
        return int(self.data.get('total_chunks_pushed', 0))

    @property
    def docs_done(self) -> int:
        return len(self._processed_set)


_VIETNAMESE_CHAR_RE = re.compile(
    r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ'
    r'òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
    r'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨ'
    r'ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]'
)


class _MLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return '\n'.join(self._parts)


def _strip_html(raw: str) -> str:
    if not raw:
        return ''
    if '<' not in raw:
        return html.unescape(raw)
    stripper = _MLStripper()
    try:
        stripper.feed(raw)
        text = stripper.get_text()
    except Exception:
        text = re.sub(r'<[^>]+>', ' ', raw)
    return html.unescape(text).strip()


@dataclass
class PipelineStats:
    total_seen: int = 0
    skipped_resume: int = 0
    passed_filter: int = 0
    docs_with_chunks: int = 0
    parents_generated: int = 0
    chunks_generated: int = 0
    chunks_pushed: int = 0

    rejected_no_content: int = 0
    rejected_corrupted: int = 0
    rejected_too_short: int = 0
    rejected_too_long: int = 0
    rejected_few_lines: int = 0
    rejected_legal_type: int = 0
    rejected_too_old: int = 0

    def log_summary(self, cfg: PipelineConfig) -> None:
        total_rejected = (
            self.rejected_no_content + self.rejected_corrupted +
            self.rejected_too_short + self.rejected_too_long +
            self.rejected_few_lines + self.rejected_legal_type +
            self.rejected_too_old
        )
        log.info('=' * 64)
        log.info('Pipeline summary')
        log.info('  total_seen            : %d', self.total_seen)
        log.info('  passed_filter         : %d', self.passed_filter)
        log.info('  docs_with_chunks      : %d', self.docs_with_chunks)
        log.info('  parents_generated     : %d', self.parents_generated)
        log.info('  chunks_generated      : %d', self.chunks_generated)
        log.info('  chunks_pushed         : %d', self.chunks_pushed)
        log.info('  total_rejected        : %d', total_rejected)
        log.info('  skip_before_year      : %d', cfg.skip_before_year)
        log.info('  avg_children_per_parent: %.1f',
                 self.chunks_generated / max(self.parents_generated, 1))
        log.info('=' * 64)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _extract_year(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    m = re.search(r'/(\d{4})$', str(date_str).strip())
    return int(m.group(1)) if m else None


def _count_meaningful_lines(text: str) -> int:
    return sum(
        1 for line in text.splitlines()
        if len(line.strip()) > 20 and not re.fullmatch(r'[#\-=_\s|]+', line.strip())
    )


def _is_likely_corrupted(text: str) -> bool:
    if not text or len(text) < 10:
        return True
    if re.search(r'[^\w\s]{20,}', text):
        return True
    pipe_ratio = text.count('|') / max(len(text), 1)
    if pipe_ratio > 0.10:
        return True
    if len(_VIETNAMESE_CHAR_RE.findall(text)) < 5:
        return True
    return False


def _stable_parent_cid(doc_id: int, parent_index: int) -> int:
    """Stable integer ID for a Parent Chunk, namespaced to avoid collision with child IDs."""
    raw = f'{doc_id}:parent:{parent_index}'.encode('utf-8')
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, byteorder='big', signed=False) & ((1 << 63) - 1)


def _stable_child_cid(doc_id: int, parent_index: int, child_index: int) -> int:
    """Stable integer ID for a Child Chunk, unique within its parent."""
    raw = f'{doc_id}:child:{parent_index}:{child_index}'.encode('utf-8')
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, byteorder='big', signed=False) & ((1 << 63) - 1)


# Regex matching Vietnamese legal article boundary markers (Dieu, Khoan, Muc, Chuong).
_ARTICLE_BOUNDARY = re.compile(
    r'(?=\n(?:Điều\s+\d+[a-z]?|Khoản\s+\d+|Mục\s+\d+|Chương\s+[IVXLCDM\d]+)\b)',
    re.UNICODE,
)


def _build_parents_from_articles(content: str, parent_size_chars: int) -> list[str]:
    """
    Split document text on legal Article boundaries (Dieu/Khoan/Muc/Chuong).
    Small articles are batched together up to parent_size_chars without ever
    splitting a single article across two parents.

    This implements the 'Logical Legal Splitting' decision from ADR-0003:
    parent boundaries must align with Article (Dieu) units.
    """
    parts = _ARTICLE_BOUNDARY.split(content)
    articles = [p.strip() for p in parts if p and p.strip()]
    if not articles:
        articles = [content.strip()]

    parents: list[str] = []
    current: list[str] = []
    current_len = 0

    for article in articles:
        article_len = len(article)
        if current and current_len + article_len > parent_size_chars:
            parents.append('\n\n'.join(current))
            current = [article]
            current_len = article_len
        else:
            if current:                # account for '\n\n' separator added on join
                current_len += 2
            current.append(article)
            current_len += article_len

    if current:
        parents.append('\n\n'.join(current))

    return parents


def _split_parent_into_children(
    parent_text: str,
    child_size_chars: int,
    child_overlap_chars: int,
    min_child_chars: int,
) -> list[str]:
    """
    Slice a Parent Chunk into overlapping Child Chunks.
    Uses a 260-char overlap (~75 Vietnamese tokens) as resolved in the
    grilling session (parent_child_grilling_decisions.md, Question 5).
    """
    children: list[str] = []
    start = 0
    text_len = len(parent_text)

    while start < text_len:
        end = min(start + child_size_chars, text_len)
        if end < text_len:
            for sep in ['\n\n', '\n', '. ', ' ']:
                cut = parent_text.rfind(sep, start + child_size_chars // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        child = parent_text[start:end].strip()
        if len(child) >= min_child_chars:
            children.append(child)
        if end >= text_len:
            break
        start = max(start + 1, end - child_overlap_chars)

    return children


def chunk_document_parent_child(
    doc_id: int,
    content: str,
    metadata: dict[str, Any],
    cfg: PipelineConfig,
) -> tuple[list[dict[str, Any]], int]:
    """
    Produce Child Chunk records for a document.

    Each record contains:
      - cid        : Child Chunk ID (Qdrant point.id)
      - parent_id  : Parent Chunk ID (FK to parents.sqlite)
      - text       : Child text (embedded by BGE-M3, scored by reranker)
      - parent_text: Full parent text (written to parents.sqlite; NOT in Qdrant)
      - metadata fields (title, url, legal_type — written to both parent and child tables)

    Returns (child_records, parent_count).
    """
    parents = _build_parents_from_articles(content, cfg.parent_size_chars)

    out: list[dict[str, Any]] = []
    meta_fields = {
        'document_number': metadata.get('document_number') or '',
        'title': metadata.get('title') or '',
        'legal_type': metadata.get('legal_type') or '',
        'legal_sectors': metadata.get('legal_sectors') or '',
        'issuing_authority': metadata.get('issuing_authority') or '',
        'issuance_date': metadata.get('issuance_date') or '',
        'url': metadata.get('url') or '',
    }

    for p_idx, parent_text in enumerate(parents):
        parent_id = _stable_parent_cid(doc_id, p_idx)
        children = _split_parent_into_children(
            parent_text,
            cfg.child_size_chars,
            cfg.child_overlap_chars,
            cfg.min_child_chars,
        )
        for c_idx, child_text in enumerate(children):
            child_id = _stable_child_cid(doc_id, p_idx, c_idx)
            out.append({
                'cid': child_id,
                'parent_id': parent_id,
                'doc_id': doc_id,
                'parent_index': p_idx,
                'child_index': c_idx,
                'text': child_text,
                'parent_text': parent_text,
                **meta_fields,
            })

    return out, len(parents)


def _download_config(cfg: PipelineConfig) -> DownloadConfig:
    return DownloadConfig(
        cache_dir=cfg.download_cache_dir,
        token=cfg.hf_token,
        max_retries=cfg.max_download_retries,
    )


def load_metadata_map(cfg: PipelineConfig) -> dict[int, dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    log.info('Downloading metadata parquet for %s', cfg.hf_dataset_id)
    file_path = hf_hub_download(
        repo_id=cfg.hf_dataset_id,
        filename=f'{cfg.hf_split}/{cfg.hf_config_metadata}.parquet',
        repo_type='dataset',
        cache_dir=cfg.download_cache_dir,
        token=cfg.hf_token,
    )
    parquet_file = pq.ParquetFile(file_path)
    meta_map: dict[int, dict[str, Any]] = {}
    for batch in tqdm(parquet_file.iter_batches(batch_size=10000), desc='Metadata batches'):
        for row in batch.to_pylist():
            doc_id = _safe_int(row.get('id'))
            if doc_id is None:
                continue
            meta_map[doc_id] = {
                'document_number': row.get('so_ky_hieu') or '',
                'title': row.get('title') or '',
                'legal_type': row.get('loai_van_ban') or '',
                'legal_sectors': row.get('linh_vuc') or '',
                'issuing_authority': row.get('co_quan_ban_hanh') or '',
                'issuance_date': row.get('ngay_ban_hanh') or '',
                'url': row.get('nguon_thu_thap') or '',
            }
    log.info('Metadata loaded: %d rows', len(meta_map))
    return meta_map


def stream_content_rows(cfg: PipelineConfig) -> Iterator[dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    file_path = hf_hub_download(
        repo_id=cfg.hf_dataset_id,
        filename=f'{cfg.hf_split}/{cfg.hf_config_content}.parquet',
        repo_type='dataset',
        cache_dir=cfg.download_cache_dir,
        token=cfg.hf_token,
    )
    parquet_file = pq.ParquetFile(file_path)
    idx = 0
    for batch in parquet_file.iter_batches(batch_size=1000):
        for row in batch.to_pylist():
            if cfg.max_content_docs > 0 and idx >= cfg.max_content_docs:
                return
            if cfg.hf_content_field != 'content' and cfg.hf_content_field in row:
                row['content'] = _strip_html(row[cfg.hf_content_field] or '')
            elif 'content' in row:
                row['content'] = _strip_html(row['content'] or '')
            yield row
            idx += 1


def pass_filters(content: str, metadata: dict[str, Any], cfg: PipelineConfig, stats: PipelineStats) -> bool:
    if not content:
        stats.rejected_no_content += 1
        return False
    if _is_likely_corrupted(content):
        stats.rejected_corrupted += 1
        return False
    if len(content) < cfg.min_content_chars:
        stats.rejected_too_short += 1
        return False
    if len(content) > cfg.max_content_chars:
        stats.rejected_too_long += 1
        return False
    if _count_meaningful_lines(content) < cfg.min_meaningful_lines:
        stats.rejected_few_lines += 1
        return False
    legal_type = str(metadata.get('legal_type') or '')
    if any(skip in legal_type for skip in cfg.skip_legal_types):
        stats.rejected_legal_type += 1
        return False
    issuance_date = str(metadata.get('issuance_date') or '')
    year = _extract_year(issuance_date)
    if year is not None and year < cfg.skip_before_year:
        stats.rejected_too_old += 1
        return False
    stats.passed_filter += 1
    return True


class DEk21DenseEmbedder:
    """Dense-only embedder using huyydangg/DEk21_hcmute_embedding (768-dim, RoBERTa).

    The DEk21 model expects pre-segmented Vietnamese input (ViTokenizer format).
    We segment with underthesea.word_tokenize + join with underscores before encoding.
    Max sequence length is 258 tokens.
    """

    def __init__(self, cfg: PipelineConfig):
        log.info('Loading DEk21 embedding model: %s', cfg.embedding_model)
        from sentence_transformers import SentenceTransformer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = SentenceTransformer(
            cfg.embedding_model,
            device=device,
        )
        self._batch_size = cfg.embedding_batch_size
        log.info('DEk21 model loaded (batch_size=%d, dim=768, device=%s)', self._batch_size, device)

    @staticmethod
    def _segment_vietnamese(text: str) -> str:
        """Segment Vietnamese text using pyvi."""
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to dense vectors (768-dim). Returns list of float lists."""
        if not texts:
            return []
        segmented = [self._segment_vietnamese(t) for t in texts]
        embeddings = self._model.encode(
            segmented,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]



class SqliteWriter:
    """Writes parent-child chunks to two separate SQLite files.

    - parents.sqlite:  one row per parent chunk (deduplicated), stores parent_text + metadata.
    - children_XXXX.sqlite: one row per child chunk, stores child text + vectors + parent_id FK.
      No parent_text column — eliminates the ~6x duplication.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.sqlite_output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Parents DB (single file, deduplicated) ──
        self._parents_path = self.out_dir / 'parents.sqlite'
        self._parents_conn = sqlite3.connect(str(self._parents_path))
        self._create_parents_table()
        self._parent_ids_written: set[int] = set()
        # Load already-written parent IDs for resume support
        try:
            for row in self._parents_conn.execute('SELECT parent_id FROM parents'):
                self._parent_ids_written.add(row[0])
        except sqlite3.OperationalError:
            pass

        # ── Children DB (sharded) ──
        self._children_conn = None
        self._current_db_idx = 0
        self._current_rows = 0
        self._init_next_children_db()

    # ── Parents table ──────────────────────────────────────────────

    def _create_parents_table(self):
        self._parents_conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS parents (
                parent_id       INTEGER PRIMARY KEY,
                doc_id          INTEGER,
                parent_text     TEXT,
                document_number TEXT,
                title           TEXT,
                legal_type      TEXT,
                legal_sectors   TEXT,
                issuing_authority TEXT,
                issuance_date   TEXT,
                url             TEXT
            )
            '''
        )
        self._parents_conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_parents_doc_id ON parents(doc_id)'
        )
        self._parents_conn.commit()

    # ── Children tables (sharded) ──────────────────────────────────

    def _init_next_children_db(self):
        if self._children_conn:
            self._children_conn.close()
        while True:
            db_path = self.out_dir / f'children_{self._current_db_idx:04d}.sqlite'
            if not db_path.exists():
                break
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute('SELECT COUNT(*) FROM children')
                count = cur.fetchone()[0]
                if count < self.cfg.sqlite_rows_per_db:
                    self._current_rows = count
                    self._children_conn = conn
                    self._create_children_table()
                    return
                conn.close()
            except sqlite3.OperationalError:
                pass
            self._current_db_idx += 1
        db_path = self.out_dir / f'children_{self._current_db_idx:04d}.sqlite'
        self._children_conn = sqlite3.connect(db_path)
        self._current_rows = 0
        self._create_children_table()

    def _create_children_table(self):
        assert self._children_conn is not None
        self._children_conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS children (
                cid             INTEGER PRIMARY KEY,
                parent_id       INTEGER,
                doc_id          INTEGER,
                parent_index    INTEGER,
                child_index     INTEGER,
                text            TEXT,
                title           TEXT,
                url             TEXT,
                legal_type      TEXT,
                dense_vector    BLOB
            )
            '''
        )
        self._children_conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_children_parent_id ON children(parent_id)'
        )
        self._children_conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_children_doc_id ON children(doc_id)'
        )
        self._children_conn.commit()

    # ── Insert ─────────────────────────────────────────────────────

    def _write_parent(self, chunk: dict) -> None:
        """INSERT OR IGNORE a parent row (deduplicated)."""
        parent_id = int(chunk['parent_id'])
        if parent_id in self._parent_ids_written:
            return
        self._parents_conn.execute(
            '''
            INSERT OR IGNORE INTO parents (
                parent_id, doc_id, parent_text,
                document_number, title, legal_type, legal_sectors,
                issuing_authority, issuance_date, url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                parent_id,
                int(chunk['doc_id']),
                str(chunk.get('parent_text') or ''),
                str(chunk.get('document_number') or ''),
                str(chunk.get('title') or ''),
                str(chunk.get('legal_type') or ''),
                str(chunk.get('legal_sectors') or ''),
                str(chunk.get('issuing_authority') or ''),
                str(chunk.get('issuance_date') or ''),
                str(chunk.get('url') or ''),
            ),
        )
        self._parent_ids_written.add(parent_id)

    def _write_child(self, chunk: dict, dense: list[float]) -> None:
        """INSERT OR REPLACE a child row (text + dense vector, no parent_text)."""
        if self._current_rows >= self.cfg.sqlite_rows_per_db:
            self._init_next_children_db()
        assert self._children_conn is not None
        self._children_conn.execute(
            '''
            INSERT OR REPLACE INTO children (
                cid, parent_id, doc_id, parent_index, child_index,
                text, title, url, legal_type,
                dense_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                int(chunk['cid']),
                int(chunk['parent_id']),
                int(chunk['doc_id']),
                int(chunk.get('parent_index', 0)),
                int(chunk.get('child_index', 0)),
                str(chunk['text']),
                str(chunk.get('title') or ''),
                str(chunk.get('url') or ''),
                str(chunk.get('legal_type') or ''),
                struct.pack(f'{len(dense)}f', *dense),
            ),
        )
        self._current_rows += 1

    def insert_points(
        self,
        chunks: list[dict],
        dense_vecs: list[list[float]],
    ) -> int:
        for chunk, dense in zip(chunks, dense_vecs):
            try:
                self._write_parent(chunk)
                self._write_child(chunk, dense)
            except Exception as exc:
                log.error('Error inserting chunk cid=%s: %s', chunk['cid'], exc)
        self._parents_conn.commit()
        if self._children_conn:
            self._children_conn.commit()
        return len(chunks)


def _flush_pending(
    pending_chunks: list[dict[str, Any]],
    pending_doc_ids: set[int],
    embedder: DEk21DenseEmbedder,
    writer: SqliteWriter,
    checkpoint: Checkpoint,
    cfg: PipelineConfig,
) -> tuple[int, int]:
    if not pending_chunks:
        return 0, 0
    texts = [c['text'] for c in pending_chunks]
    dense_vecs = embedder.encode(texts)
    pushed = writer.insert_points(pending_chunks, dense_vecs)
    docs_committed = len(pending_doc_ids)
    checkpoint.mark_done(pending_doc_ids, chunks_added=pushed)
    pending_chunks.clear()
    pending_doc_ids.clear()
    return pushed, docs_committed


def run_pipeline(cfg: PipelineConfig) -> PipelineStats:
    stats = PipelineStats()
    checkpoint = Checkpoint(cfg.checkpoint_file)
    if cfg.resume and not cfg.dry_run:
        checkpoint.load()

    log.info('Step 1/4 - Load metadata')
    metadata_map = load_metadata_map(cfg)

    embedder: Optional[DEk21DenseEmbedder] = None
    writer: Optional[SqliteWriter] = None

    if not cfg.dry_run:
        log.info('Step 2/4 - Load embedder and SQLite writer')
        embedder = DEk21DenseEmbedder(cfg)
        writer = SqliteWriter(cfg)

    log.info('Step 3/4 - Stream content, filter, chunk (parent-child)')
    pending_chunks: list[dict[str, Any]] = []
    pending_doc_ids: set[int] = set()
    docs_since_checkpoint = 0
    progress_total = cfg.max_content_docs if cfg.max_content_docs > 0 else None

    for row in tqdm(stream_content_rows(cfg), total=progress_total, desc='Content', unit='doc'):
        stats.total_seen += 1
        doc_id = _safe_int(row.get('id'))
        if doc_id is None:
            continue
        if not cfg.dry_run and checkpoint.is_done(doc_id):
            stats.skipped_resume += 1
            continue

        content = str(row.get('content') or '').strip()
        meta = metadata_map.get(doc_id, {})

        if not pass_filters(content, meta, cfg, stats):
            if not cfg.dry_run:
                checkpoint.mark_done([doc_id], chunks_added=0)
                docs_since_checkpoint += 1
            continue

        chunks, parent_count = chunk_document_parent_child(doc_id, content, meta, cfg)
        if not chunks:
            if not cfg.dry_run:
                checkpoint.mark_done([doc_id], chunks_added=0)
                docs_since_checkpoint += 1
            continue

        stats.docs_with_chunks += 1
        stats.parents_generated += parent_count
        stats.chunks_generated += len(chunks)

        if cfg.dry_run:
            continue

        pending_chunks.extend(chunks)
        pending_doc_ids.add(doc_id)

        if len(pending_chunks) >= cfg.flush_chunks_threshold:
            assert embedder is not None and writer is not None
            pushed, docs_committed = _flush_pending(pending_chunks, pending_doc_ids, embedder, writer, checkpoint, cfg)
            stats.chunks_pushed += pushed
            docs_since_checkpoint += docs_committed

        if stats.total_seen % cfg.log_every == 0:
            log.info(
                'Progress seen=%d passed=%d parents=%d children_gen=%d children_push=%d pending=%d',
                stats.total_seen, stats.passed_filter,
                stats.parents_generated, stats.chunks_generated,
                stats.chunks_pushed, len(pending_chunks),
            )

        if not cfg.dry_run and docs_since_checkpoint >= cfg.checkpoint_every_docs:
            checkpoint.save()
            docs_since_checkpoint = 0
            log.info('Checkpoint saved: docs_done=%d chunks_pushed=%d', checkpoint.docs_done, checkpoint.total_chunks_pushed)

    if not cfg.dry_run and pending_chunks:
        assert embedder is not None and writer is not None
        pushed, _ = _flush_pending(pending_chunks, pending_doc_ids, embedder, writer, checkpoint, cfg)
        stats.chunks_pushed += pushed

    if not cfg.dry_run:
        checkpoint.save()
        stats.chunks_pushed = checkpoint.total_chunks_pushed

    log.info('Step 4/4 - Summary')
    stats.log_summary(cfg)

    if writer is not None:
        if writer._parents_conn is not None:
            writer._parents_conn.close()
        if writer._children_conn is not None:
            writer._children_conn.close()
        log.info('Finished. Output dir: %s', cfg.sqlite_output_dir)

    return stats


if __name__ == '__main__':
    print('Parent-Child Pipeline definitions loaded.')

    cfg.resume = False
    cfg.dry_run = False

    # Smoke-test with limited docs first:
    # cfg.max_content_docs = 500
    # cfg.dry_run = True

    stats = run_pipeline(cfg)
    print('Done. SQLite output directory:', cfg.sqlite_output_dir)

    out_dir = Path(cfg.sqlite_output_dir)

    parents_file = out_dir / 'parents.sqlite'
    if parents_file.exists():
        conn = sqlite3.connect(parents_file)
        parent_count = conn.execute('SELECT COUNT(*) FROM parents').fetchone()[0]
        conn.close()
        print(f'  parents.sqlite: {parent_count:,} unique parents')

    child_files = sorted(out_dir.glob('children_*.sqlite'))
    print(f'Children SQLite files generated: {len(child_files)}')
    for f in child_files[:10]:
        conn = sqlite3.connect(f)
        row_count = conn.execute('SELECT COUNT(*) FROM children').fetchone()[0]
        parent_count = conn.execute('SELECT COUNT(DISTINCT parent_id) FROM children').fetchone()[0]
        conn.close()
        print(f'  {f.name}: {row_count:,} children, {parent_count:,} unique parents')

