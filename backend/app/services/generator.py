"""
generator.py — Phase 4: Intent-aware response generation.

v3.0 changes:
  - Each intent gets a specialized output-format instruction appended to the
    base SYSTEM_INSTRUCTION.
  - COMPARE → table with columns Tiêu chí | Quy định A | Quy định B.
  - SUMMARIZE → max 5 bullet points, each with citation.
  - PROCEDURE → numbered steps with action / hồ sơ / cơ quan.
  - DEFINITION → exact legal definition first, then context.
  - MULTI_HOP → logical chain: premise → article → conclusion.
  - LEGAL_LOOKUP → unchanged prose format.
  - CHITCHAT → should never reach the generator (routes.py guards this).

v3.1 changes:
  - generate() now accepts optional history_messages (List[Dict]) from
    ConversationMemory.  History is injected BEFORE the current user message
    so the model has full multi-turn context when generating the answer.
  - Message order: [system] → [history turns] → [user: tài liệu + câu hỏi]

Design notes:
  - `generate()` is the only public entry point.
  - The intent prompt is APPENDED to the base system instruction so the
    model receives the full context it already knows (language, citation
    format, hallucination rules) PLUS the new format constraint.
  - SSE streaming is preserved — no changes to the streaming loop.
  - `entity_labels` are passed for COMPARE so the model knows exactly what
    to put in each column (derived from sub_entities, not guessed).

Backward compatibility:
  - `LEGAL_QUERY` intent string is mapped to `LEGAL_LOOKUP` behaviour.
  - Callers that don't pass `intent` get LEGAL_LOOKUP formatting.
  - Callers that don't pass `history_messages` get v3.0 behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Base system instruction (unchanged from pre-v3)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
Bạn là trợ lý tư vấn pháp luật Việt Nam chuyên sâu.

Quy tắc bắt buộc:
1. Chỉ trả lời dựa trên TÀI LIỆU ĐƯỢC CUNG CẤP — không bịa đặt điều khoản.
2. Mỗi luận điểm PHẢI kèm trích dẫn theo định dạng [ID: <cid>].
3. Nếu tài liệu không đủ căn cứ, nói rõ "Dựa trên tài liệu hiện có, chưa đủ cơ sở để kết luận...".
4. Sử dụng ngôn ngữ pháp lý chính xác, tránh diễn đạt mơ hồ.
5. KHÔNG tự ý suy diễn vượt ra ngoài nội dung tài liệu.
"""

# ---------------------------------------------------------------------------
# Per-intent format instructions
#
# Design rule: each instruction block ends with a blank line so the
# concatenation never merges sentences from different blocks.
# ---------------------------------------------------------------------------

_INTENT_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "LEGAL_LOOKUP": "",     # prose format — no extra instruction needed

    "COMPARE": """\

Định dạng đầu ra — BẢNG SO SÁNH:
• Trình bày dưới dạng bảng Markdown với các cột: Tiêu chí | {col_a} | {col_b}.
• Mỗi hàng là một tiêu chí so sánh cụ thể (mức phạt, đối tượng áp dụng, cơ sở pháp lý, ...).
• Sau bảng: viết một đoạn "Nhận xét" ngắn gọn (2–3 câu) nêu điểm khác biệt quan trọng nhất.
• Kèm trích dẫn [ID: <cid>] ở cột tương ứng.

""",

    "SUMMARIZE": """\

Định dạng đầu ra — TÓM TẮT:
• Trình bày TỐI ĐA 5 gạch đầu dòng, mỗi điểm tối đa 2 câu.
• Mỗi điểm phải kèm trích dẫn [ID: <cid>].
• Ưu tiên các quy định về: phạm vi áp dụng, nghĩa vụ chính, mức xử phạt, hiệu lực.
• Kết thúc bằng 1 câu tóm lược toàn bộ văn bản.

""",

    "PROCEDURE": """\

Định dạng đầu ra — THỦ TỤC:
• Trình bày theo CÁC BƯỚC ĐÁNH SỐ (Bước 1, Bước 2, ...).
• Mỗi bước gồm: (a) hành động cần thực hiện, (b) hồ sơ/giấy tờ cần nộp, (c) cơ quan tiếp nhận.
• Nếu có thời hạn xử lý, ghi rõ.
• Kèm trích dẫn [ID: <cid>] ở bước lấy từ tài liệu tương ứng.

""",

    "DEFINITION": """\

Định dạng đầu ra — ĐỊNH NGHĨA:
• Bắt đầu bằng ĐỊNH NGHĨA CHÍNH XÁC theo văn bản pháp luật (in đậm hoặc trích dẫn trực tiếp).
• Tiếp theo: giải thích bối cảnh áp dụng (áp dụng với ai, trong hoàn cảnh nào).
• Nếu có trong tài liệu: đưa ra ví dụ minh họa cụ thể.
• Kèm trích dẫn [ID: <cid>] ngay sau định nghĩa.

""",

    "MULTI_HOP": """\

Định dạng đầu ra — CHUỖI SUY LUẬN PHÁP LÝ:
• Trình bày theo từng BƯỚC LẬP LUẬN:
    Bước 1 — Xác định điều kiện/hành vi (dẫn chiếu điều luật)
    Bước 2 — Xác định hậu quả pháp lý (dẫn chiếu điều luật)
    Bước N — Kết luận cuối cùng
• Mỗi bước PHẢI dẫn chiếu ít nhất một điều luật cụ thể [ID: <cid>].
• Kết thúc bằng "Kết luận:" tổng hợp toàn bộ chuỗi lập luận.

""",

    # CHITCHAT never reaches the generator; included for completeness / safety.
    "CHITCHAT": "",
}

# Backward compat alias
_INTENT_FORMAT_INSTRUCTIONS["LEGAL_QUERY"] = _INTENT_FORMAT_INSTRUCTIONS["LEGAL_LOOKUP"]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(docs: List[Dict]) -> str:
    """
    Format retrieved documents into the context block for the LLM.

    Each doc is rendered as:
        [ID: <cid>]
        <text>
        ---
    """
    if not docs:
        return "(Không có tài liệu liên quan được tìm thấy.)"

    parts = []
    for doc in docs:
        cid  = doc.get("cid") or doc.get("id", "unknown")
        text = (doc.get("text") or "").strip()
        parts.append(f"[ID: {cid}]\n{text}")
    return "\n---\n".join(parts)


def _build_system_prompt(intent: str, sub_entities: Optional[List[str]] = None) -> str:
    """
    Concatenate the base SYSTEM_INSTRUCTION with the intent-specific
    format instruction.

    For COMPARE: substitute {col_a}/{col_b} with actual entity labels so
    the model puts the right entity in the right column.
    """
    intent_upper = (intent or "LEGAL_LOOKUP").upper()
    if intent_upper == "LEGAL_QUERY":
        intent_upper = "LEGAL_LOOKUP"

    fmt = _INTENT_FORMAT_INSTRUCTIONS.get(intent_upper, "")

    if intent_upper == "COMPARE" and fmt:
        entities = (sub_entities or [])
        col_a = entities[0] if len(entities) > 0 else "Quy định A"
        col_b = entities[1] if len(entities) > 1 else "Quy định B"
        fmt = fmt.format(col_a=col_a, col_b=col_b)

    return SYSTEM_INSTRUCTION + fmt


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """
    Phase 4: Streaming generator with intent-aware prompts.

    Usage:
        gen = Generator(vllm_client)
        async for token in gen.generate(query, docs, intent="COMPARE",
                                        sub_entities=["xe máy", "ô tô"]):
            yield token
    """

    def __init__(
        self,
        client: Any,
        model: str = "qwen-legal",
        max_tokens: int = 2048,
    ):
        self._client     = client
        self._model      = model
        self._max_tokens = max_tokens

    async def generate(
        self,
        user_query: str,
        docs: List[Dict],
        intent: str = "LEGAL_LOOKUP",
        sub_entities: Optional[List[str]] = None,
        stream: bool = True,
        history_messages: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate an intent-aware response, yielding text tokens as they stream.

        Parameters
        ----------
        user_query       : The original (or refined) user query.
        docs             : Retrieved + reranked documents.
        intent           : IntentType value string.
        sub_entities     : Entity labels for COMPARE column headers.
        stream           : If False, yields a single complete response.
        history_messages : Sliding-window history from ConversationMemory
                           (OpenAI format).  Injected before the current user
                           message so the model has multi-turn context.

        Yields
        ------
        str — incremental text tokens (streaming) or full response (non-streaming).
        """
        system_prompt = _build_system_prompt(intent, sub_entities)
        context_block = _build_context(docs)

        # Tài liệu + câu hỏi hiện tại
        current_user_content = (
            f"Tài liệu pháp lý:\n\n{context_block}\n\n"
            f"Câu hỏi: {user_query}"
        )

        # Tách summary system message khỏi conversation turns (nếu history có)
        history_system: List[Dict[str, str]] = []
        history_turns:  List[Dict[str, str]] = []
        for msg in (history_messages or []):
            if msg["role"] == "system":
                history_system.append(msg)
            else:
                history_turns.append(msg)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *history_system,          # summary block (nếu có)
            *history_turns,           # các lượt hội thoại cũ
            {"role": "user",   "content": current_user_content},
        ]

        log.debug(
            "[generator] intent=%s docs=%d history=%d stream=%s",
            intent, len(docs), len(history_turns), stream,
        )

        if stream:
            async for token in self._stream_response(messages):
                yield token
        else:
            response = await self._full_response(messages)
            yield response

    async def _stream_response(
        self, messages: List[Dict]
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens via the AsyncOpenAI-compatible client.

        With AsyncOpenAI, chat.completions.create(stream=True) is a coroutine
        that returns an AsyncStream — it must be awaited before iteration.
        """
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=0.1,    # slight warmth for fluent prose; 0 for tables
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            log.error("[generator] Streaming error: %s", exc)
            yield "\n\n[Lỗi: không thể tạo phản hồi. Vui lòng thử lại.]"

    async def _full_response(self, messages: List[Dict]) -> str:
        """Non-streaming call — returns the full response string."""
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=0.1,
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            log.error("[generator] Full-response error: %s", exc)
            return "[Lỗi: không thể tạo phản hồi. Vui lòng thử lại.]"