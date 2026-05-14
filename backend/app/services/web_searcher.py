"""
Tavily Web Search Fallback Service for Legal RAG.

Activated ONLY when the local Qdrant vector DB fails to return sufficiently
relevant documents (mean rerank score below threshold).  All searches are
restricted to a configurable allow-list of trusted Vietnamese legal domains.

Design decisions:
  • Singleflight cache — prevents Thundering Herd (N concurrent identical
    queries only trigger 1 Tavily API call; the rest await the same Future).
  • Circuit breaker — after consecutive failures, Tavily is temporarily
    bypassed to avoid cascading latency on a degraded external API.
  • Sync SDK wrapped in asyncio.to_thread — tavily-python is synchronous;
    offloading to a thread keeps the FastAPI event loop unblocked.
  • Smart truncation — content is cut at the nearest sentence boundary
    rather than mid-word, preserving legal clause integrity.
"""

import asyncio
import hashlib
import logging
import re
import time
from typing import Dict, List, Optional

from cachetools import TTLCache

from app.core.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _cache_key(query: str) -> str:
    """Deterministic, case-insensitive cache key."""
    return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()


def _deterministic_cid(url: str) -> str:
    """Stable CID from URL so LLM citations are reproducible across requests."""
    return f"web_{hashlib.md5(url.encode('utf-8')).hexdigest()[:8]}"


def _smart_truncate(text: str, max_chars: int) -> str:
    """Truncate at the nearest sentence boundary (. ? !) before max_chars.

    Falls back to newline, then hard cut if no sentence end is found.
    This preserves full legal clauses instead of cutting mid-Điều.
    """
    if len(text) <= max_chars:
        return text

    # Search for last sentence-end within budget
    window = text[:max_chars]

    # Try sentence boundary first (Vietnamese legal text ends with . or :)
    last_sentence = max(
        window.rfind(". "),
        window.rfind(".\n"),
        window.rfind("? "),
        window.rfind(":\n"),
    )
    if last_sentence > max_chars * 0.4:  # only if we keep at least 40%
        return window[: last_sentence + 1].rstrip()

    # Fall back to paragraph boundary
    last_newline = window.rfind("\n")
    if last_newline > max_chars * 0.4:
        return window[:last_newline].rstrip()

    # Hard cut + ellipsis
    return window.rstrip() + "…"


# ──────────────────────────────────────────────
# Core Service
# ──────────────────────────────────────────────

class TavilyLegalSearcher:
    """Production-grade Tavily integration with singleflight cache and circuit breaker."""

    def __init__(self) -> None:
        self._client = None  # Lazy init — only created when API key is present

        # ── Cache (TTL-based, bounded) ──
        self._cache: TTLCache = TTLCache(
            maxsize=settings.TAVILY_CACHE_MAX_SIZE,
            ttl=settings.TAVILY_CACHE_TTL_SEC,
        )

        # ── Singleflight: per-key in-flight Futures ──
        # Prevents Thundering Herd — only the first request for a given query
        # calls Tavily; concurrent duplicates await the same Future.
        self._inflight: Dict[str, asyncio.Future] = {}
        self._inflight_lock = asyncio.Lock()

        # ── Circuit breaker (mirrors query_reflector.py pattern) ──
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    # ------------------------------------------------------------------
    # Lazy client init
    # ------------------------------------------------------------------

    def _ensure_client(self) -> bool:
        """Initialise TavilyClient only when first needed."""
        if not settings.TAVILY_API_KEY:
            return False
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=settings.TAVILY_API_KEY)
                logger.info("✅ TavilyClient initialised successfully.")
            except Exception as exc:
                logger.error("Failed to initialise TavilyClient: %s", exc)
                return False
        return True

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _is_circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= settings.TAVILY_CIRCUIT_FAIL_THRESHOLD:
            self._circuit_open_until = (
                time.monotonic() + settings.TAVILY_CIRCUIT_COOLDOWN_SEC
            )
            logger.warning(
                "[WebSearch] Circuit OPEN for %.0fs after %d consecutive failures.",
                settings.TAVILY_CIRCUIT_COOLDOWN_SEC,
                self._consecutive_failures,
            )
            self._consecutive_failures = 0

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    # ------------------------------------------------------------------
    # Sync search (runs in thread)
    # ------------------------------------------------------------------

    def _search_sync(self, query: str) -> list:
        """Blocking Tavily API call — must be offloaded to a thread."""
        return self._client.search(
            query=query,
            search_depth="advanced",
            include_domains=settings.TAVILY_ALLOWED_DOMAINS,
            max_results=settings.TAVILY_MAX_RESULTS,
            include_raw_content=True,
        )

    # ------------------------------------------------------------------
    # Result parsing & validation
    # ------------------------------------------------------------------

    def _parse_results(self, raw_response: dict) -> List[Dict]:
        """
        Convert Tavily JSON to a list of pipeline-compatible dicts.

        Each dict has the shape:
            {
              "cid":      str,          # deterministic hash of URL
              "text":     str,          # full content prefixed with source URL
              "score":    float,        # Tavily relevance score (0–1)
              "metadata": {
                "source":   str,        # original URL
                "title":    str,        # page title
                "from_web": True,
              },
            }

        This dict shape is consumed by:
          - routes._format_sources()  (reads cid, score, metadata.source, metadata.title)
          - generator._build_context() (reads cid, text)
          - routes._should_fallback()  (reads score)
        """
        docs: List[Dict] = []
        results = raw_response.get("results", [])

        for item in results:
            # Prefer raw_content (full page) over snippet
            content = (
                item.get("raw_content")
                or item.get("content")
                or ""
            ).strip()

            # Validation gate — skip empty / trivially short pages
            if len(content) < 100:
                logger.debug(
                    "[WebSearch] Skipping result with %d chars (< 100): %s",
                    len(content),
                    item.get("url", "?")[:80],
                )
                continue

            # Smart truncation to stay within context window budget
            content = _smart_truncate(content, settings.TAVILY_MAX_CONTENT_CHARS)

            url   = item.get("url", "")
            title = item.get("title", "")
            cid   = _deterministic_cid(url)

            # Use Tavily's own relevance score (0–1), not a fake 1.0
            tavily_score = float(item.get("score", 0.0))

            docs.append({
                "cid":   cid,
                "text":  f"[Nguồn: {url}]\n{content}",
                "score": round(tavily_score, 4),
                "metadata": {
                    "source":   url,
                    "title":    title,
                    "from_web": True,
                },
            })

        logger.info(
            "[WebSearch] Parsed %d valid docs from %d raw results.",
            len(docs),
            len(results),
        )
        return docs

    # ------------------------------------------------------------------
    # Public async API — singleflight + cache + circuit breaker
    # ------------------------------------------------------------------

    async def search_trusted_laws(self, query: str) -> List[Dict]:
        """Search trusted legal domains via Tavily. Results are cached.

        Singleflight semantics: if an identical query is already in-flight,
        this call awaits the existing Future instead of issuing a duplicate
        API request (prevents Thundering Herd).
        """
        # ── Guard: no API key or circuit open → silent empty return ──
        if not self._ensure_client():
            logger.warning("[WebSearch] TAVILY_API_KEY not set. Skipping web search.")
            return []

        if self._is_circuit_open():
            logger.warning("[WebSearch] Circuit is OPEN. Skipping web search.")
            return []

        ck = _cache_key(query)

        # ── Fast path: cache hit (no lock needed, TTLCache is atomic for reads) ──
        cached = self._cache.get(ck)
        if cached is not None:
            logger.info("[WebSearch] Cache HIT for query (key=%s).", ck[:8])
            return cached

        # ── Singleflight: leader/follower coordination ──
        is_leader = False
        async with self._inflight_lock:
            # Double-check cache after acquiring lock
            cached = self._cache.get(ck)
            if cached is not None:
                return cached

            if ck in self._inflight:
                # Another coroutine is already fetching → we are a follower
                existing_future = self._inflight[ck]
            else:
                # We are the leader — create a future for followers to await
                existing_future = None
                leader_future: asyncio.Future = asyncio.get_event_loop().create_future()
                self._inflight[ck] = leader_future
                is_leader = True

        # ── Follower path: just await the leader's result ──
        if not is_leader:
            logger.info("[WebSearch] Singleflight AWAIT for key=%s.", ck[:8])
            try:
                return await existing_future
            except Exception:
                return []

        # ── Leader path: execute actual API call ──
        try:
            start_t = time.perf_counter()
            raw_response = await asyncio.wait_for(
                asyncio.to_thread(self._search_sync, query),
                timeout=settings.TAVILY_TIMEOUT_SEC,
            )
            elapsed = time.perf_counter() - start_t
            logger.info("[WebSearch] Tavily API responded in %.2fs.", elapsed)

            docs = self._parse_results(raw_response)
            self._record_success()

            # Populate cache
            self._cache[ck] = docs

            # Resolve the future so followers get the result
            leader_future.set_result(docs)
            return docs

        except asyncio.TimeoutError:
            logger.error(
                "[WebSearch] Tavily timed out after %.1fs.",
                settings.TAVILY_TIMEOUT_SEC,
            )
            self._record_failure()
            leader_future.set_result([])
            return []

        except Exception as exc:
            logger.error("[WebSearch] Tavily call failed: %s", exc, exc_info=True)
            self._record_failure()
            leader_future.set_result([])
            return []

        finally:
            # Clean up inflight tracker
            async with self._inflight_lock:
                self._inflight.pop(ck, None)


# ──────────────────────────────────────────────
# Module-level singleton (matches retriever_engine / generator_engine pattern)
# ──────────────────────────────────────────────
web_searcher_engine = TavilyLegalSearcher()
