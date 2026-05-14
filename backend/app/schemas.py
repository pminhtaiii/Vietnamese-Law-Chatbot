"""
schemas.py — Pydantic models for API request/response validation.

v3.0 changes:
  - ChatResponse gains an `intent` field (str, default "LEGAL_LOOKUP").
    This lets the frontend render intent-appropriate UI:
      COMPARE   → table layout
      PROCEDURE → numbered-step layout
      DEFINITION → definition callout box
      others    → default prose
  - IntentType enum is re-exported here so the frontend SDK only needs to
    import from schemas (single source of truth for contract consumers).
  - No breaking changes: `intent` defaults to "LEGAL_LOOKUP" so existing
    clients that don't inspect the field continue working.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# Re-export the enum so API clients import from one place.
from app.services.query_reflector import IntentType   # noqa: F401


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096,
                         description="User query (Vietnamese legal question or chitchat).")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID for multi-turn context tracking.",
    )
    stream: bool = Field(
        default=True,
        description="If true, the response is sent as an SSE stream.",
    )
    include_content: bool = Field(
        default=False,
        description="If true, includes the raw text content in the DocumentSource.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve (1–20). Capped to avoid abuse.",
    )


class RetrieveRequest(BaseModel):
    """Request model for the retrieval-only endpoint POST /api/retrieve."""
    message: str = Field(..., min_length=1, max_length=4096,
                         description="User query to retrieve documents for.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve (1–20).",
    )
    include_content: bool = Field(
        default=True,
        description="If true, includes the raw text content in each source document.",
    )


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class DocumentSource(BaseModel):
    """A single retrieved source document reference."""
    id:     str   = Field(default="",  description="Document chunk ID (cid).")
    title:  str   = Field(default="",  description="Document title or section heading.")
    url:    str   = Field(default="",  description="Original URL (web fallback docs only).")
    score:  float = Field(default=0.0, description="Reranked relevance score (0–1).")
    entity: str   = Field(
        default="",
        description=(
            "For COMPARE results: the entity this document was retrieved for "
            "(e.g. 'xe máy').  Empty for non-COMPARE intents."
        ),
    )
    content: Optional[str] = Field(
        default=None,
        description="Raw text content of the document chunk. Omitted by default to save bandwidth.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    """
    Non-streaming response body.

    For streaming responses the HTTP layer sends SSE events instead; this
    model is used for the final consolidated object returned after the stream
    completes, and for the /chat/sync endpoint.
    """
    answer:  str                = Field(..., description="Generated answer text.")
    sources: List[DocumentSource] = Field(
        default_factory=list,
        description="Retrieved source documents used to generate the answer.",
    )
    intent: str = Field(
        default="LEGAL_LOOKUP",
        description=(
            "Detected intent type.  One of: LEGAL_LOOKUP, COMPARE, SUMMARIZE, "
            "PROCEDURE, DEFINITION, MULTI_HOP, CHITCHAT."
        ),
    )
    confidence: float = Field(
        default=0.0,
        description="Intent classification confidence score (0–1).",
    )
    from_web: bool = Field(
        default=False,
        description="True if at least one source came from Tavily web fallback.",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "3.0.0"


class MetricsResponse(BaseModel):
    """Snapshot of in-memory pipeline metrics."""
    total_requests:    int = 0
    legal_requests:    int = 0
    chitchat_requests: int = 0
    web_fallbacks:     int = 0
    errors:            int = 0
    by_intent: dict = Field(default_factory=dict)