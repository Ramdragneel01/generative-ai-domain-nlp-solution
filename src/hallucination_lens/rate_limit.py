
"""In-memory request rate limiting helpers for API protection."""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import perf_counter


class InMemoryRateLimiter:
    """Sliding-window limiter keyed by client identifier."""

    def __init__(self, window_seconds: int = 60) -> None:
        """Initialize limiter store for per-client request timestamps."""

        self._window_seconds = window_seconds
        self._store: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, limit: int) -> bool:
        """Return whether key can perform another request in current time window."""

        now = perf_counter()
        with self._lock:
            queue = self._store.setdefault(key, deque())
            cutoff = now - self._window_seconds
            while queue and queue[0] < cutoff:
                queue.popleft()

            if len(queue) >= limit:
                return False

            queue.append(now)
        return True
