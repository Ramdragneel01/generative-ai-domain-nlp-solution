
"""Runtime settings for hallucination-lens API service."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os


def _parse_float_env(name: str, default: str) -> float:
    """Parse float environment value and fail fast on invalid runtime configuration."""

    value = os.getenv(name, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid float") from exc


def _parse_int_env(name: str, default: str) -> int:
    """Parse integer environment value and fail fast on invalid runtime configuration."""

    value = os.getenv(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid integer") from exc


def _parse_bool_env(name: str, default: str) -> bool:
    """Parse common boolean string representations from environment variables."""

    value = os.getenv(name, default).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value (true/false)")


def _parse_csv_env(name: str, default: str) -> list[str]:
    """Parse comma-separated environment values into a normalized non-empty list."""

    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


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
    max_request_bytes: int
    cors_origins: list[str]
    trusted_hosts: list[str]
    enable_gzip: bool
    gzip_minimum_size: int
    preload_model_on_startup: bool
    enable_hsts: bool
    api_key: str


def _validate_settings(settings: Settings) -> None:
    """Validate production safety bounds and fail fast on invalid startup settings."""

    if settings.min_threshold < 0.0 or settings.min_threshold > 1.0:
        raise ValueError("MIN_THRESHOLD must be between 0 and 1")
    if settings.max_threshold < 0.0 or settings.max_threshold > 1.0:
        raise ValueError("MAX_THRESHOLD must be between 0 and 1")
    if settings.min_threshold > settings.max_threshold:
        raise ValueError("MIN_THRESHOLD must not exceed MAX_THRESHOLD")
    if settings.default_threshold < settings.min_threshold:
        raise ValueError("FAITHFULNESS_THRESHOLD must be >= MIN_THRESHOLD")
    if settings.default_threshold > settings.max_threshold:
        raise ValueError("FAITHFULNESS_THRESHOLD must be <= MAX_THRESHOLD")
    if settings.max_batch_items <= 0:
        raise ValueError("MAX_BATCH_ITEMS must be greater than zero")
    if settings.max_context_chars <= 0:
        raise ValueError("MAX_CONTEXT_CHARS must be greater than zero")
    if settings.max_response_chars <= 0:
        raise ValueError("MAX_RESPONSE_CHARS must be greater than zero")
    if settings.rate_limit_per_minute <= 0:
        raise ValueError("RATE_LIMIT_PER_MINUTE must be greater than zero")
    if settings.max_request_bytes <= 0:
        raise ValueError("MAX_REQUEST_BYTES must be greater than zero")
    if settings.gzip_minimum_size < 0:
        raise ValueError("GZIP_MINIMUM_SIZE must be zero or greater")
    if not settings.cors_origins:
        raise ValueError("CORS_ORIGINS must include at least one origin")
    if not settings.trusted_hosts:
        raise ValueError("TRUSTED_HOSTS must include at least one host")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment variables using safe production defaults."""

    settings = Settings(
        app_name=os.getenv("APP_NAME", "hallucination-lens"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        model_name=os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        default_threshold=_parse_float_env("FAITHFULNESS_THRESHOLD", "0.6"),
        min_threshold=_parse_float_env("MIN_THRESHOLD", "0.3"),
        max_threshold=_parse_float_env("MAX_THRESHOLD", "0.9"),
        max_batch_items=_parse_int_env("MAX_BATCH_ITEMS", "50"),
        max_context_chars=_parse_int_env("MAX_CONTEXT_CHARS", "50000"),
        max_response_chars=_parse_int_env("MAX_RESPONSE_CHARS", "50000"),
        rate_limit_per_minute=_parse_int_env("RATE_LIMIT_PER_MINUTE", "120"),
        max_request_bytes=_parse_int_env("MAX_REQUEST_BYTES", "1048576"),
        cors_origins=_parse_csv_env(
            "CORS_ORIGINS",
            "http://127.0.0.1:4176,http://localhost:4176",
        ),
        trusted_hosts=_parse_csv_env("TRUSTED_HOSTS", "127.0.0.1,localhost,testserver"),
        enable_gzip=_parse_bool_env("ENABLE_GZIP", "true"),
        gzip_minimum_size=_parse_int_env("GZIP_MINIMUM_SIZE", "1024"),
        preload_model_on_startup=_parse_bool_env("PRELOAD_MODEL_ON_STARTUP", "false"),
        enable_hsts=_parse_bool_env("ENABLE_HSTS", "false"),
        api_key=os.getenv("API_KEY", "").strip(),
    )
    _validate_settings(settings)
    return settings
