from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    
    # Qdrant configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "http://qdrant:6333")
    QDRANT_DB_PATH: str = str(BASE_DIR / "data" / "qdrant_db_m3")
    COLLECTION_NAME: str = "vietnamese_laws_m3"
    
    # AI models configuration
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "thanhtantran/Vietnamese_Reranker"
    RERANKER_MAX_LENGTH: int = 2304
    # Runtime max_length for tokenization during rerank requests.
    # Much shorter than model capability (2304) because legal chunks are
    # typically 300-800 chars.  Reduces tokenization RAM by ~4.5x.
    RERANKER_REQUEST_MAX_LENGTH: int = int(os.getenv("RERANKER_REQUEST_MAX_LENGTH", "512"))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")
    QUERY_REFLECT_MODEL: str = os.getenv("QUERY_REFLECT_MODEL", "gemini-3.1-flash-lite-preview")
    USE_FP16: bool = True

    # Query reflection + intent routing configuration
    QUERY_REFLECT_TIMEOUT_SEC: float = float(os.getenv("QUERY_REFLECT_TIMEOUT_SEC", "2.5"))
    QUERY_REFLECT_HISTORY_TURNS: int = int(os.getenv("QUERY_REFLECT_HISTORY_TURNS", "4"))
    QUERY_REFLECT_MAX_MSG_CHARS: int = int(os.getenv("QUERY_REFLECT_MAX_MSG_CHARS", "420"))
    QUERY_REFLECT_MAX_HISTORY_CHARS: int = int(os.getenv("QUERY_REFLECT_MAX_HISTORY_CHARS", "1400"))
    QUERY_REFLECT_CIRCUIT_FAIL_THRESHOLD: int = int(os.getenv("QUERY_REFLECT_CIRCUIT_FAIL_THRESHOLD", "5"))
    QUERY_REFLECT_CIRCUIT_COOLDOWN_SEC: float = float(os.getenv("QUERY_REFLECT_CIRCUIT_COOLDOWN_SEC", "60"))
    CHITCHAT_CONFIDENCE_THRESHOLD: float = float(os.getenv("CHITCHAT_CONFIDENCE_THRESHOLD", "0.75"))
    ROUTING_METRICS_LOG_EVERY_N: int = int(os.getenv("ROUTING_METRICS_LOG_EVERY_N", "25"))
    
    # ── GraphRAG configuration ────────────────────────────────────
    GRAPHRAG_ENABLED: bool = os.getenv("GRAPHRAG_ENABLED", "true").lower() in ("true", "1", "yes")
    GRAPHRAG_LANCEDB_PATH: str = os.getenv(
        "GRAPHRAG_LANCEDB_PATH",
        str(BASE_DIR.parent / "data_pipelines" / "graph_rag" / "output" / "lancedb"),
    )
    GRAPHRAG_ENTITIES_PARQUET: str = os.getenv(
        "GRAPHRAG_ENTITIES_PARQUET",
        str(BASE_DIR.parent / "data_pipelines" / "graph_rag" / "output" / "entities.parquet"),
    )
    GRAPHRAG_RELATIONSHIPS_PARQUET: str = os.getenv(
        "GRAPHRAG_RELATIONSHIPS_PARQUET",
        str(BASE_DIR.parent / "data_pipelines" / "graph_rag" / "output" / "relationships.parquet"),
    )

    # API keys
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', "")
    COHERE_API_KEY: str = os.getenv('COHERE_API_KEY', "")
    TAVILY_API_KEY: str = os.getenv('TAVILY_API_KEY', "")

    # ── Tavily Web Search Fallback ──────────────────────────────
    # Domain allow-list — only trusted Vietnamese legal sources.
    TAVILY_ALLOWED_DOMAINS: list = [
        "thuvienphapluat.vn",
        "luatvietnam.vn",
    ]
    TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
    TAVILY_MAX_CONTENT_CHARS: int = int(os.getenv("TAVILY_MAX_CONTENT_CHARS", "2000"))
    TAVILY_TIMEOUT_SEC: float = float(os.getenv("TAVILY_TIMEOUT_SEC", "8.0"))
    TAVILY_CACHE_TTL_SEC: int = int(os.getenv("TAVILY_CACHE_TTL_SEC", "3600"))
    TAVILY_CACHE_MAX_SIZE: int = int(os.getenv("TAVILY_CACHE_MAX_SIZE", "256"))

    # If the MEAN rerank score of all retrieved docs falls below this,
    # trigger Tavily web search as fallback.
    # Note: local reranker (sigmoid) produces bimodal scores (~0 or ~1),
    # so 0.15 rarely triggers when local reranker is active. Cohere scores
    # are more gradual. Tune this if switching between rerankers.
    RAG_FALLBACK_SCORE_THRESHOLD: float = float(os.getenv("RAG_FALLBACK_SCORE_THRESHOLD", "0.05"))

    # Circuit breaker — stop calling Tavily after N consecutive failures
    # for a cooldown period, preventing cascading latency.
    TAVILY_CIRCUIT_FAIL_THRESHOLD: int = int(os.getenv("TAVILY_CIRCUIT_FAIL_THRESHOLD", "3"))
    TAVILY_CIRCUIT_COOLDOWN_SEC: float = float(os.getenv("TAVILY_CIRCUIT_COOLDOWN_SEC", "120.0"))

    # Pydantic config
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
settings = Settings()
    
    
    