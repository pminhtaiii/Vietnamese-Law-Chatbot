"""
conversation_memory.py — Quản lý lịch sử hội thoại nhiều lượt.

Cung cấp một sliding-window memory với cơ chế tóm tắt tự động khi
lịch sử quá dài, đảm bảo LLM luôn nhận được context phù hợp mà không
vượt quá giới hạn token.

Kiến trúc:
  ┌─────────────────────────────────────────────────────────────────┐
  │  ConversationMemory (per conversation_id)                        │
  │                                                                  │
  │  full_history : List[Turn]   — toàn bộ lịch sử raw             │
  │  summary      : str | None   — tóm tắt các turn cũ hơn window  │
  │                                                                  │
  │  get_context_messages(window=5) →                               │
  │    [summary_msg?, ...last_N_turns_as_OAI_messages]              │
  └─────────────────────────────────────────────────────────────────┘

Luồng xử lý:
  1. Mỗi request gọi memory.get_context_messages() để lấy history ngắn gọn.
  2. Sau khi có câu trả lời, gọi memory.add_turn(user_q, assistant_a).
  3. Nếu len(full_history) > SUMMARIZE_THRESHOLD, bắt đầu chu kỳ tóm tắt:
     - Tóm tắt N turn cũ nhất bằng LLM.
     - Giữ lại summary + window turn gần nhất.

Thiết kế nhất quán với claude.md:
  - Toàn bộ async/await.
  - Không mix prompt logic với memory logic.
  - Singleton store per conversation_id (dict-based, in-memory).
    → Thay bằng Redis/DB cho production multi-instance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Tunable constants (có thể override qua env trong config.py nếu cần)
# ---------------------------------------------------------------------------

# Số lượt trao đổi (user+assistant = 1 lượt) đưa vào LLM.
WINDOW_SIZE: int = 5

# Khi tổng số lượt vượt ngưỡng này thì kích hoạt tóm tắt.
# = WINDOW_SIZE + số lượt muốn tóm tắt một lần.
SUMMARIZE_THRESHOLD: int = WINDOW_SIZE + 3   # tóm tắt khi có > 8 lượt

# Số lượt cũ nhất đem đi tóm tắt mỗi lần (batch).
SUMMARIZE_BATCH: int = 3

# Giới hạn ký tự mỗi turn khi nhét vào summary prompt (tránh prompt quá dài).
_MAX_TURN_CHARS: int = 800


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """Một lượt hội thoại: user hỏi → assistant trả lời."""
    user:      str
    assistant: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationState:
    """Trạng thái đầy đủ của một conversation_id."""
    turns:   List[Turn] = field(default_factory=list)
    summary: str        = ""          # tóm tắt các turn cũ đã bị nén
    _lock:   asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


# ---------------------------------------------------------------------------
# System prompt cho việc tóm tắt lịch sử
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = """\
Bạn là trợ lý tóm tắt lịch sử hội thoại pháp luật.

Nhiệm vụ: Tóm tắt NGẮN GỌN các lượt hội thoại dưới đây thành 3–5 câu.
Chỉ giữ lại: chủ đề pháp lý đã bàn, kết luận chính, và ngữ cảnh quan trọng.
Không thêm thông tin ngoài cuộc hội thoại.
Trả lời bằng tiếng Việt, không dùng markdown.
"""


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

# conversation_id → ConversationState
_STORE: Dict[str, ConversationState] = {}
_STORE_LOCK = asyncio.Lock()


async def _get_or_create(conversation_id: str) -> ConversationState:
    async with _STORE_LOCK:
        if conversation_id not in _STORE:
            _STORE[conversation_id] = ConversationState()
        return _STORE[conversation_id]


# ---------------------------------------------------------------------------
# ConversationMemory (public API)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Quản lý lịch sử hội thoại cho một conversation_id cụ thể.

    Cách dùng:
        mem = ConversationMemory(conversation_id, llm_client, model)

        # Trước khi gọi pipeline:
        history_messages = await mem.get_context_messages()

        # Sau khi có câu trả lời:
        await mem.add_turn(user_query, assistant_answer)
    """

    def __init__(
        self,
        conversation_id: str,
        llm_client: Optional[Any] = None,
        model: str = "gemini-2.0-flash",
        window_size: int = WINDOW_SIZE,
    ):
        self._cid    = conversation_id
        self._client = llm_client
        self._model  = model
        self._window = window_size

    # ------------------------------------------------------------------
    # Public: get context for LLM
    # ------------------------------------------------------------------

    async def get_context_messages(self) -> List[Dict[str, str]]:
        """
        Trả về danh sách messages (OpenAI format) đại diện cho lịch sử
        hội thoại để nhét vào LLM prompt.

        Cấu trúc trả về:
          [
            {"role": "system", "content": "<summary nếu có>"},  # optional
            {"role": "user",   "content": "<turn N-k+1 user>"},
            {"role": "assistant", "content": "<turn N-k+1 assistant>"},
            ...
            {"role": "user",   "content": "<turn N user>"},
            {"role": "assistant", "content": "<turn N assistant>"},
          ]

        Lưu ý: câu hỏi hiện tại của user KHÔNG có trong danh sách này —
        caller sẽ append nó vào sau.
        """
        state = await _get_or_create(self._cid)

        messages: List[Dict[str, str]] = []

        # Thêm summary (nếu có) dưới dạng system message phụ
        if state.summary:
            messages.append({
                "role":    "system",
                "content": f"[Tóm tắt hội thoại trước]\n{state.summary}",
            })

        # Thêm window N lượt gần nhất
        recent = state.turns[-self._window:] if state.turns else []
        for turn in recent:
            messages.append({"role": "user",      "content": turn.user})
            messages.append({"role": "assistant",  "content": turn.assistant})

        return messages

    # ------------------------------------------------------------------
    # Public: record a completed turn
    # ------------------------------------------------------------------

    async def add_turn(self, user_query: str, assistant_answer: str) -> None:
        """
        Ghi nhận một lượt hội thoại đã hoàn thành.

        Nếu tổng số lượt vượt SUMMARIZE_THRESHOLD, tự động kích hoạt
        tóm tắt batch bất đồng bộ (fire-and-forget nếu muốn, hoặc await).
        """
        state = await _get_or_create(self._cid)

        async with state._lock:
            state.turns.append(Turn(user=user_query, assistant=assistant_answer))

            # Kích hoạt tóm tắt nếu đủ ngưỡng
            if len(state.turns) >= SUMMARIZE_THRESHOLD and self._client:
                await self._summarize_old_turns(state)

    # ------------------------------------------------------------------
    # Public: clear conversation
    # ------------------------------------------------------------------

    async def clear(self) -> None:
        """Xoá toàn bộ lịch sử của conversation này."""
        async with _STORE_LOCK:
            _STORE.pop(self._cid, None)

    # ------------------------------------------------------------------
    # Internal: summarize
    # ------------------------------------------------------------------

    async def _summarize_old_turns(self, state: ConversationState) -> None:
        """
        Tóm tắt SUMMARIZE_BATCH lượt cũ nhất và cập nhật state.summary.

        Gọi từ add_turn() sau khi đã giữ state._lock.
        Nếu LLM call thất bại, giữ nguyên state (không mất data).
        """
        batch = state.turns[:SUMMARIZE_BATCH]
        remaining = state.turns[SUMMARIZE_BATCH:]

        # Dựng nội dung cần tóm tắt
        history_text = ""
        if state.summary:
            history_text += f"[Tóm tắt cũ hơn]\n{state.summary}\n\n"
        history_text += "[Các lượt cần tóm tắt thêm]\n"
        for i, t in enumerate(batch, 1):
            u_text = t.user[:_MAX_TURN_CHARS]
            a_text = t.assistant[:_MAX_TURN_CHARS]
            history_text += f"Lượt {i}:\n  Người dùng: {u_text}\n  Trợ lý: {a_text}\n"

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                    {"role": "user",   "content": history_text},
                ],
                max_tokens=300,
                temperature=0.0,
                stream=False,
            )
            new_summary = (resp.choices[0].message.content or "").strip()
            if new_summary:
                state.summary = new_summary
                state.turns   = remaining
                log.info(
                    "[memory] Summarized %d turns for cid=%s → %d chars",
                    len(batch), self._cid, len(new_summary),
                )
        except Exception as exc:
            # Nếu tóm tắt thất bại, không xoá data gốc — chỉ log warning.
            log.warning(
                "[memory] Summarization failed for cid=%s: %s (data preserved)",
                self._cid, exc,
            )


# ---------------------------------------------------------------------------
# Module-level factory / singleton store
# ---------------------------------------------------------------------------

# Singleton map: conversation_id → ConversationMemory instance
# Dùng để tái sử dụng ConversationMemory object giữa các request.
_MEMORY_INSTANCES: Dict[str, ConversationMemory] = {}
_MEMORY_LOCK = asyncio.Lock()


async def get_memory(
    conversation_id: str,
    llm_client: Optional[Any] = None,
    model: str = "gemini-2.0-flash",
    window_size: int = WINDOW_SIZE,
) -> ConversationMemory:
    """
    Trả về ConversationMemory singleton cho conversation_id.

    Tạo mới nếu chưa tồn tại.  Thread-safe với asyncio.Lock.

    Parameters
    ----------
    conversation_id : ID duy nhất của cuộc hội thoại.
    llm_client      : AsyncOpenAI client để tóm tắt (có thể None — tắt tóm tắt).
    model           : Model dùng để tóm tắt.
    window_size     : Số lượt gần nhất nhét vào context.
    """
    async with _MEMORY_LOCK:
        if conversation_id not in _MEMORY_INSTANCES:
            _MEMORY_INSTANCES[conversation_id] = ConversationMemory(
                conversation_id=conversation_id,
                llm_client=llm_client,
                model=model,
                window_size=window_size,
            )
        return _MEMORY_INSTANCES[conversation_id]
