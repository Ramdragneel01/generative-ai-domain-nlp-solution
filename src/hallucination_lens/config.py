
"""Runtime settings for hallucination-lens API service."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    """Represents runtime configuration for API, validation, and governance limits."""

    app_name: str
    app_version: str
    model_name: str
    default_threshold: float
    min_threshold: float
    max_threshold: float
    max_batch_items: int
    max_context_chars: int
    max_response_chars: int
    rate_limit_per_minute: int
    cors_origins: list[str]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment variables using safe production defaults."""

    origin_value = os.getenv("CORS_ORIGINS", "http://127.0.0.1:4176,http://localhost:4176")
    return Settings(
        app_name=os.getenv("APP_NAME", "hallucination-lens"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        model_name=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        default_threshold=float(os.getenv("FAITHFULNESS_THRESHOLD", "0.6")),
        min_threshold=float(os.getenv("MIN_THRESHOLD", "0.3")),
        max_threshold=float(os.getenv("MAX_THRESHOLD", "0.9")),
        max_batch_items=int(os.getenv("MAX_BATCH_ITEMS", "50")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "50000")),
        max_response_chars=int(os.getenv("MAX_RESPONSE_CHARS", "50000")),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")),
        cors_origins=[origin.strip() for origin in origin_value.split(",") if origin.strip()],
    )
