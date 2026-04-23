
"""Validation helpers for CLI and library inputs."""


def validate_text_argument(name: str, value: str, min_length: int = 5, max_length: int = 20000) -> str:
    """Validate and normalize a text input before scoring."""

    normalized = value.strip()
    if len(normalized) < min_length:
        raise ValueError(f"{name} must be at least {min_length} characters")
    if len(normalized) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    return normalized


def validate_threshold(value: float) -> float:
    """Ensure threshold is in closed interval [0, 1]."""

    if value < 0.0 or value > 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return value
