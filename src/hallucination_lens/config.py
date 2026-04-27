
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
    rate_limit_backend: str
    redis_url: str
    redis_socket_timeout_seconds: float
    rate_limit_redis_key_prefix: str
    max_request_bytes: int
    cors_origins: list[str]
    trusted_hosts: list[str]
    enable_gzip: bool
    gzip_minimum_size: int
    preload_model_on_startup: bool
    enable_hsts: bool
    gateway_auth_enabled: bool
    gateway_auth_header: str
    gateway_auth_secret: str
    gateway_principal_header: str
    require_gateway_principal: bool
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
    if settings.rate_limit_backend not in {"memory", "redis"}:
        raise ValueError("RATE_LIMIT_BACKEND must be either memory or redis")
    if settings.redis_socket_timeout_seconds <= 0:
        raise ValueError("REDIS_SOCKET_TIMEOUT_SECONDS must be greater than zero")
    if settings.rate_limit_backend == "redis" and not settings.redis_url:
        raise ValueError("REDIS_URL must be configured when RATE_LIMIT_BACKEND is redis")
    if not settings.rate_limit_redis_key_prefix:
        raise ValueError("RATE_LIMIT_REDIS_KEY_PREFIX must not be empty")
    if settings.max_request_bytes <= 0:
        raise ValueError("MAX_REQUEST_BYTES must be greater than zero")
    if settings.gzip_minimum_size < 0:
        raise ValueError("GZIP_MINIMUM_SIZE must be zero or greater")
    if not settings.cors_origins:
        raise ValueError("CORS_ORIGINS must include at least one origin")
    if not settings.trusted_hosts:
        raise ValueError("TRUSTED_HOSTS must include at least one host")
    if not settings.gateway_auth_header:
        raise ValueError("GATEWAY_AUTH_HEADER must not be empty")
    if not settings.gateway_principal_header:
        raise ValueError("GATEWAY_PRINCIPAL_HEADER must not be empty")
    if settings.gateway_auth_enabled and not settings.gateway_auth_secret:
        raise ValueError("GATEWAY_AUTH_SECRET must be set when GATEWAY_AUTH_ENABLED is true")
    if settings.require_gateway_principal and not settings.gateway_auth_enabled:
        raise ValueError("REQUIRE_GATEWAY_PRINCIPAL requires GATEWAY_AUTH_ENABLED=true")


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
        rate_limit_backend=os.getenv("RATE_LIMIT_BACKEND", "memory").strip().lower(),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0").strip(),
        redis_socket_timeout_seconds=_parse_float_env("REDIS_SOCKET_TIMEOUT_SECONDS", "1.0"),
        rate_limit_redis_key_prefix=os.getenv(
            "RATE_LIMIT_REDIS_KEY_PREFIX",
            "hallucination-lens:rate-limit",
        ).strip(),
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
        gateway_auth_enabled=_parse_bool_env("GATEWAY_AUTH_ENABLED", "false"),
        gateway_auth_header=os.getenv("GATEWAY_AUTH_HEADER", "X-Gateway-Auth").strip(),
        gateway_auth_secret=os.getenv("GATEWAY_AUTH_SECRET", "").strip(),
        gateway_principal_header=os.getenv("GATEWAY_PRINCIPAL_HEADER", "X-Principal-Id").strip(),
        require_gateway_principal=_parse_bool_env("REQUIRE_GATEWAY_PRINCIPAL", "false"),
        api_key=os.getenv("API_KEY", "").strip(),
    )
    _validate_settings(settings)
    return settings
