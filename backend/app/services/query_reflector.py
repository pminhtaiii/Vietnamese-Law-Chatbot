"""
query_reflector.py — Phase 2: Intent routing and query expansion.

v3.0 changes:
  - Expanded IntentType from 2 → 7 intents (+ LEGAL_QUERY alias for compat).
  - Added sub_entities field to ReflectionResult for COMPARE / MULTI_HOP.
  - Added rule-based fast-paths (regex) for COMPARE, SUMMARIZE, PROCEDURE,
    DEFINITION — reduces LLM calls for obvious patterns.
  - Updated LLM system prompt to route into the full 7-intent taxonomy and
    to populate sub_entities for compound queries.
  - LEGAL_QUERY is kept as an alias of LEGAL_LOOKUP so existing log pipelines
    and metrics code don't break.

v3.1 changes:
  - reflect() now accepts optional history_messages (List[Dict]) — a
    sliding-window of prior conversation turns in OpenAI message format.
  - History is injected between the system prompt and the current user query
    so the LLM can resolve coreferences ("luật đó", "điều khoản trên", ...).
  - Rule-based fast-path is NOT history-aware (by design — rules fire only
    on strong syntactic signals in the current query; history doesn't change
    whether we see "so sánh" or "thủ tục").

Intent taxonomy:
  LEGAL_LOOKUP  — direct legal fact retrieval (was LEGAL_QUERY)
  COMPARE       — compare 2+ provisions / laws / penalties
  SUMMARIZE     — condense a decree or chapter
  PROCEDURE     — step-by-step legal process
  DEFINITION    — define a legal term / concept
  MULTI_HOP     — multi-step legal reasoning
  CHITCHAT      — non-legal conversation

Logical note on fast-paths:
  Vietnamese text may appear with or without diacritics (many mobile users type
  without). Both forms are covered below by using Unicode-normalised variants.
  Patterns are applied BEFORE the LLM call to reduce latency and API cost.
  The LLM remains the authoritative router for ambiguous cases.

Backward compatibility:
  IntentType.LEGAL_QUERY == IntentType.LEGAL_LOOKUP  →  True
  (achieved via the class-level alias; serialisation always emits "LEGAL_LOOKUP")
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    LEGAL_LOOKUP = "LEGAL_LOOKUP"
    COMPARE      = "COMPARE"
    SUMMARIZE    = "SUMMARIZE"
    PROCEDURE    = "PROCEDURE"
    DEFINITION   = "DEFINITION"
    MULTI_HOP    = "MULTI_HOP"
    CHITCHAT     = "CHITCHAT"

# Backward-compat alias — code that checks `intent == IntentType.LEGAL_QUERY`
# continues to work without any changes to the call-sites.
IntentType.LEGAL_QUERY = IntentType.LEGAL_LOOKUP   # type: ignore[attr-defined]

# Convenience: all non-chitchat intents (used in routing and metrics).
LEGAL_INTENTS: frozenset[IntentType] = frozenset(
    IntentType(v)
    for v in IntentType.__members__
    if v != "CHITCHAT"
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReflectionResult:
    intent: IntentType
    refined_query: str
    confidence: float
    response: str = ""
    source: str = "llm"           # "llm" | "rule" | "fallback"
    reason: str = ""
    expanded_queries: list = field(default_factory=list)
    # For COMPARE / MULTI_HOP: individual entities/sub-questions to retrieve.
    # e.g. COMPARE "xe máy vs ô tô" → ["nồng độ cồn xe máy", "nồng độ cồn ô tô"]
    # e.g. MULTI_HOP → ["sub-question 1", "sub-question 2"]
    sub_entities: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule-based fast-path patterns
#
# Each pattern list targets one intent.  Patterns cover both full-diacritic
# Vietnamese and common diacritic-stripped input (normalized to NFC then
# ASCII-folded for the match, but we keep the original query for retrieval).
#
# Design choice: false-positive cost is low (LLM would have agreed) but
# false-negative cost is also low (falls through to LLM).  Keep patterns
# conservative — only match strong signals.
# ---------------------------------------------------------------------------

_VN_BASE_MAP = str.maketrans("đĐ", "dD")


def _normalize(text: str) -> str:
    """
    Fold diacritics and lower-case for pattern matching only.

    Two-step:
      1. Map Vietnamese atomic base-letter substitutions (đ→d, Đ→D) that
         NFD decomposition cannot reach (they are standalone code points,
         not base-letter + combining-mark pairs).
      2. NFD decompose, then strip all combining marks (Unicode category Mn).
    """
    text = text.translate(_VN_BASE_MAP)
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


_COMPARE_PATTERNS = [
    # "so sánh", "khác nhau", "giống nhau", "khác biệt"
    r"\b(so\s+s[aá]nh|kh[aá]c\s+nhau|gi[oô]ng\s+nhau|kh[aá]c\s+bi[eê]t)\b",
    # "giữa A và B" with a law-like keyword nearby
    r"\bgi[uư]a\b.{1,80}\b(lu[aâ]t|ngh[ij]\s+[dđ][iị]nh|[dđ]i[eề]u|kho[aả]n)\b",
    # "A va B" near penalty/vehicle keywords
    r"\b(xe\s+m[aá]y|[oô]\s*t[oô]|xe\s+t[aả]i).{1,40}\b(v[aà]|hay)\b.{1,40}\b(xe\s+m[aá]y|[oô]\s*t[oô]|xe\s+t[aả]i)\b",
]

_SUMMARIZE_PATTERNS = [
    # "tóm tắt", "tổng kết", "nội dung chính", "tổng quan"
    r"\b(t[oó]m\s+t[aắ]t|t[oổ]ng\s+k[eế]t|n[oộ]i\s+dung\s+ch[ií]nh|t[oổ]ng\s+quan)\b",
]

_PROCEDURE_PATTERNS = [
    # "thủ tục", "quy trình", "cách làm", "các bước" near action verbs
    r"\b(th[uủ]\s+t[uụ]c|quy\s+tr[iì]nh|c[aá]ch\s+l[aà]m|c[aá]c\s+b[uướ]c)\b"
    r".{0,60}\b([dđ][aă]ng\s+k[yý]|xin|n[oộ]p|c[aấ]p|l[aà]m)\b",
    # Simpler: "thủ tục + noun"
    r"\b(th[uủ]\s+t[uụ]c)\b",
    r"\b(b[uướ]c\s+\d|b[uướ]c\s+th[uứ])\b",
]

_DEFINITION_PATTERNS = [
    # "là gì", "định nghĩa", "khái niệm", "hiểu thế nào"
    r"\b(l[aà]\s+g[iì]|[dđ][iị]nh\s+ngh[iĩ]a|kh[aá]i\s+ni[eệ]m|hi[eể]u\s+th[eế]\s+n[aà]o)\b",
]

# MULTI_HOP: "nếu … thì …" chains, "trong trường hợp", consequence keywords
_MULTI_HOP_PATTERNS = [
    # "nếu ... thì" conditional chains (diacritic + normalized forms)
    r"\bn[eế]u\b.{5,120}\bth[iì]\b",
    r"\bneu\b.{5,120}\bthi\b",
    # "trong trường hợp" — both full diacritic and normalized forms
    r"\btrong\s+tr[uườ]ng\s+h[oợ]p\b",
    r"\btrong\s+truong\s+hop\b",
    # consequence vocabulary near legal action vocabulary
    r"\b(h[aậ]u\s+qu[aả]|h[eệ]\s+qu[aả]|d[aẫ]n\s+[dđ][eế]n|[aả]nh\s+h[uưở][oở]ng)\b"
    r".{0,80}\b(ph[aạ]t|x[uử]\s+l[yý]|tr[aá]ch\s+nhi[eệ]m)\b",
]

# Compile all patterns once at import time.
def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]

_RE_COMPARE   = _compile(_COMPARE_PATTERNS)
_RE_SUMMARIZE = _compile(_SUMMARIZE_PATTERNS)
_RE_PROCEDURE = _compile(_PROCEDURE_PATTERNS)
_RE_DEFINITION = _compile(_DEFINITION_PATTERNS)
_RE_MULTI_HOP  = _compile(_MULTI_HOP_PATTERNS)


def _rule_based_intent(query: str) -> Optional[IntentType]:
    """
    Return an IntentType if a strong rule fires, else None (→ use LLM).

    Priority order matters: MULTI_HOP before LEGAL_LOOKUP so consequence
    chains are not treated as plain lookups.  COMPARE before DEFINITION so
    "so sánh định nghĩa" routes to COMPARE.
    """
    # Apply patterns to both original and diacritic-stripped forms.
    targets = [query, _normalize(query)]

    for t in targets:
        for p in _RE_MULTI_HOP:
            if p.search(t):
                return IntentType.MULTI_HOP
    for t in targets:
        for p in _RE_COMPARE:
            if p.search(t):
                return IntentType.COMPARE
    for t in targets:
        for p in _RE_SUMMARIZE:
            if p.search(t):
                return IntentType.SUMMARIZE
    for t in targets:
        for p in _RE_PROCEDURE:
            if p.search(t):
                return IntentType.PROCEDURE
    for t in targets:
        for p in _RE_DEFINITION:
            if p.search(t):
                return IntentType.DEFINITION

    return None  # No rule matched — fall through to LLM.


# ---------------------------------------------------------------------------
# Sub-entity extraction helpers
# ---------------------------------------------------------------------------

def _extract_compare_entities(query: str) -> list[str]:
    """
    Heuristic split of a COMPARE query into two sub-entity retrieval queries.

    Splits on connectors: "và", "với", "vs", "so với", "hay".
    Returns the full query in a list when no split point is found (so the
    retriever can still do two slightly different searches using HyDE expansion).
    """
    # Try explicit connector split
    split_re = re.compile(
        r"\s+(?:v[aà]|v[oớ]i|vs\.?|so\s+v[oớ]i|hay)\s+",
        re.IGNORECASE | re.UNICODE,
    )
    parts = split_re.split(query, maxsplit=1)
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        if a and b:
            return [a, b]

    # Fallback: return the whole query twice so the retriever still runs
    # two passes (useful with diverse HyDE expansions).
    return [query, query]


def _extract_multihop_subquestions(query: str) -> list[str]:
    """
    Decompose a MULTI_HOP query into sub-questions (best-effort heuristic).

    The LLM prompt is authoritative; this is a local fallback when the LLM
    call fails or when using the rule-based fast-path.
    """
    # Split on "nếu … thì" structure
    m = re.search(r"(n[eế]u\s+.+?)\s+th[iì]\s+(.+)", query, re.IGNORECASE | re.UNICODE)
    if m:
        condition = m.group(1).strip()
        consequence = m.group(2).strip()
        return [condition, consequence]

    # Fallback: return query as single item; chained retrieval uses it twice.
    return [query]


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
Bạn là bộ định tuyến truy vấn cho hệ thống tư vấn pháp luật Việt Nam.

Nhiệm vụ:
  1. Phân loại câu hỏi của người dùng vào đúng nhóm ý định (intent).
  2. Viết lại câu hỏi dưới dạng ngắn gọn, rõ ràng phù hợp cho tra cứu vector (refined_query).
  3. Tạo tối đa 3 câu mở rộng bổ sung (expanded_queries) — không lặp lại refined_query.
  4. Với COMPARE / MULTI_HOP: trích xuất danh sách sub_entities gồm các đối tượng / câu hỏi con.

Bảng intent:
  LEGAL_LOOKUP — Tra cứu điều luật / quy định cụ thể (mặc định cho câu hỏi pháp lý đơn giản)
  COMPARE      — So sánh 2+ quy định, mức phạt, điều luật hoặc đối tượng áp dụng
  SUMMARIZE    — Tóm tắt nội dung một văn bản / chương / điều luật
  PROCEDURE    — Hỏi thủ tục, quy trình, các bước thực hiện
  DEFINITION   — Hỏi định nghĩa / khái niệm pháp lý
  MULTI_HOP    — Câu hỏi cần suy luận nhiều bước ("Nếu A thì B?")
  CHITCHAT     — Hội thoại thông thường, không liên quan đến pháp luật

Quy tắc:
  - Chỉ trả về JSON, KHÔNG có markdown hoặc lời dẫn.
  - Với COMPARE: sub_entities = danh sách 2 đối tượng đang so sánh (ví dụ: ["xe máy", "ô tô"]).
  - Với MULTI_HOP: sub_entities = danh sách câu hỏi con theo thứ tự logic pháp lý.
  - Nếu CHITCHAT: refined_query = câu gốc, expanded_queries = [], sub_entities = [].
  - confidence: 0.0–1.0 (độ chắc chắn của phân loại).

Cấu trúc JSON:
{
  "intent": "<INTENT>",
  "refined_query": "<câu hỏi được viết lại>",
  "confidence": <float 0-1>,
  "reason": "<lý do ngắn gọn>",
  "expanded_queries": ["<query 1>", "<query 2>"],
  "sub_entities": ["<entity 1>", "<entity 2>"]
}
"""

_CHITCHAT_RESPONSE = (
    "Xin chào! Tôi là trợ lý tư vấn pháp luật Việt Nam. "
    "Tôi có thể giúp bạn tra cứu quy định pháp luật, giải thích điều khoản, "
    "hoặc hướng dẫn thủ tục pháp lý. Bạn cần hỗ trợ gì?"
)


# ---------------------------------------------------------------------------
# Main reflector class
# ---------------------------------------------------------------------------

class QueryReflector:
    """
    Phase 2: Intent routing + query rewriting.

    Usage:
        reflector = QueryReflector(llm_client)
        result = await reflector.reflect(user_query)
        # result.intent  → IntentType.COMPARE
        # result.sub_entities → ["xe máy", "ô tô"]
    """

    def __init__(self, llm_client: Any, model: str = "qwen-legal"):
        self._client = llm_client
        self._model  = model

    async def reflect(
        self,
        query: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
    ) -> ReflectionResult:
        """
        Route query to the correct intent, rewrite it, and return a
        ReflectionResult.

        Parameters
        ----------
        query            : Câu hỏi hiện tại của người dùng.
        history_messages : Lịch sử hội thoại (OpenAI format) từ ConversationMemory.
                           Được inject vào LLM call để giải quyết coreference
                           ("luật đó", "điều khoản trên", v.v.).
                           None hoặc [] → hành vi như v3.0 (không có history).

        Fast-path order:
          1. Empty / too-short queries → CHITCHAT (guard).
          2. Rule-based fast-path → immediate return without LLM call.
             (History không ảnh hưởng rule-based: patterns chỉ xét syntax.)
          3. LLM call với history injected.
          4. Parse + validate JSON; fall back to LEGAL_LOOKUP on parse error.
        """
        query = query.strip()

        # Guard: empty input
        if not query or len(query) < 3:
            return ReflectionResult(
                intent=IntentType.CHITCHAT,
                refined_query=query,
                confidence=1.0,
                response=_CHITCHAT_RESPONSE,
                source="rule",
                reason="Empty or too-short input.",
            )

        # ── Fast-path: rule-based ──
        rule_intent = _rule_based_intent(query)
        if rule_intent is not None and rule_intent != IntentType.CHITCHAT:
            sub_entities = self._derive_sub_entities(query, rule_intent)
            expanded     = self._simple_expansions(query, rule_intent)
            log.debug(
                "[reflector] Rule fast-path: %s  sub_entities=%s",
                rule_intent.value, sub_entities,
            )
            return ReflectionResult(
                intent=rule_intent,
                refined_query=query,
                confidence=0.80,        # conservative; LLM would be higher
                source="rule",
                reason=f"Matched rule-based pattern for {rule_intent.value}.",
                expanded_queries=expanded,
                sub_entities=sub_entities,
            )

        # ── LLM routing (history-aware) ──
        return await self._llm_reflect(query, history_messages or [])

    def _derive_sub_entities(self, query: str, intent: IntentType) -> list[str]:
        if intent == IntentType.COMPARE:
            return _extract_compare_entities(query)
        if intent == IntentType.MULTI_HOP:
            return _extract_multihop_subquestions(query)
        return []

    def _simple_expansions(self, query: str, intent: IntentType) -> list[str]:
        """
        Minimal expansions generated locally when bypassing the LLM.
        These are intentionally generic — the LLM expansion is richer.
        """
        if intent == IntentType.DEFINITION:
            return [f"định nghĩa pháp lý {query}", f"{query} theo bộ luật"]
        if intent == IntentType.PROCEDURE:
            return [f"hồ sơ {query}", f"cơ quan thực hiện {query}"]
        if intent == IntentType.SUMMARIZE:
            return [f"nội dung {query}", f"điều khoản chính {query}"]
        return []

    async def _llm_reflect(
        self,
        query: str,
        history_messages: List[Dict[str, str]],
    ) -> ReflectionResult:
        """
        Send the query to the LLM (with conversation history) and parse
        the structured JSON response.

        Message ordering:
          [system: routing prompt]
          [history turn 1 user] [history turn 1 assistant]
          ...
          [history turn N user] [history turn N assistant]
          [user: current query]

        History messages có thể bao gồm một system message đầu tiên từ
        ConversationMemory (summary block) — giữ nguyên thứ tự đó.
        """
        # Tách summary system msg (nếu có) ra khỏi conversation turns
        history_system: List[Dict[str, str]] = []
        history_turns:  List[Dict[str, str]] = []
        for msg in history_messages:
            if msg["role"] == "system":
                history_system.append(msg)
            else:
                history_turns.append(msg)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *history_system,          # summary block (nếu có)
            *history_turns,           # các lượt user/assistant cũ
            {"role": "user", "content": query},
        ]

        if history_messages:
            log.debug(
                "[reflector] LLM call with %d history messages",
                len(history_messages),
            )

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
                stream=False,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            log.warning("[reflector] LLM call failed: %s — falling back to LEGAL_LOOKUP", exc)
            return ReflectionResult(
                intent=IntentType.LEGAL_LOOKUP,
                refined_query=query,
                confidence=0.5,
                source="fallback",
                reason=f"LLM error: {exc}",
            )

        return self._parse_llm_response(query, raw)

    def _parse_llm_response(self, original_query: str, raw: str) -> ReflectionResult:
        """
        Parse the LLM's JSON response into a ReflectionResult.

        Graceful degradation:
          - Strips markdown fences if the LLM adds them.
          - Falls back to LEGAL_LOOKUP on JSON parse error.
          - Falls back to LEGAL_LOOKUP on unknown intent value.
          - Validates sub_entities is a list; resets to [] on bad type.
        """
        # Strip markdown fences (model sometimes wraps in ```json ... ```)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)

        try:
            obj = json.loads(clean)
        except json.JSONDecodeError as exc:
            log.warning("[reflector] JSON parse error (%s) — raw: %.200s", exc, raw)
            return ReflectionResult(
                intent=IntentType.LEGAL_LOOKUP,
                refined_query=original_query,
                confidence=0.4,
                source="fallback",
                reason=f"JSON parse error: {exc}",
            )

        # Resolve intent value
        intent_str  = obj.get("intent", "LEGAL_LOOKUP").upper()
        # LEGAL_QUERY is accepted as a legacy alias
        if intent_str == "LEGAL_QUERY":
            intent_str = "LEGAL_LOOKUP"
        try:
            intent = IntentType(intent_str)
        except ValueError:
            log.warning("[reflector] Unknown intent '%s' — defaulting to LEGAL_LOOKUP", intent_str)
            intent = IntentType.LEGAL_LOOKUP

        refined        = obj.get("refined_query", original_query).strip() or original_query
        confidence     = float(obj.get("confidence", 0.7))
        reason         = obj.get("reason", "")
        expanded       = obj.get("expanded_queries", [])
        sub_entities   = obj.get("sub_entities", [])

        # Type safety
        if not isinstance(expanded, list):
            expanded = []
        if not isinstance(sub_entities, list):
            sub_entities = []

        # For COMPARE/MULTI_HOP: if the LLM returned empty sub_entities,
        # fall back to heuristic extraction so retrieval still works.
        if intent in (IntentType.COMPARE, IntentType.MULTI_HOP) and not sub_entities:
            sub_entities = self._derive_sub_entities(original_query, intent)

        # CHITCHAT: inject a friendly response, skip retrieval
        response = ""
        if intent == IntentType.CHITCHAT:
            response = _CHITCHAT_RESPONSE

        return ReflectionResult(
            intent=intent,
            refined_query=refined,
            confidence=confidence,
            response=response,
            source="llm",
            reason=reason,
            expanded_queries=expanded[:3],       # cap at 3
            sub_entities=sub_entities[:4],       # cap at 4 to limit API cost
        )


# ---------------------------------------------------------------------------
# Convenience: is_legal_intent()
# ---------------------------------------------------------------------------

def is_legal_intent(intent: IntentType) -> bool:
    """Return True for any intent that requires retrieval (not CHITCHAT)."""
    return intent != IntentType.CHITCHAT