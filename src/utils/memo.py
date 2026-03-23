"""
memo.py — Smart memoization with automatic invalidation.

Provides a caching system that:
  - Memoizes analysis results based on input parameters
  - Automatically invalidates when inputs change significantly
  - Tracks cache statistics for monitoring
  - Supports TTL-based expiration
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from functools import wraps
from typing import Any

# Cache configuration
_CACHE_MAX_SIZE = 50
_CACHE_TTL_SECONDS = 300  # 5 minutes
_SENSITIVITY_THRESHOLD = 0.01


@dataclass
class CacheEntry:
    """A cached analysis result with metadata."""

    result: Any
    timestamp: float
    hits: int = 0
    input_hash: str = ""


class AnalysisCache:
    """
    LRU cache for analysis results with TTL and input-based invalidation.
    """

    def __init__(
        self,
        max_size: int = _CACHE_MAX_SIZE,
        ttl_seconds: float = _CACHE_TTL_SECONDS,
    ):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _hash_inputs(self, *args, **kwargs) -> str:
        """Generate a stable hash from input parameters."""
        def serialize(obj: Any) -> Any:
            if is_dataclass(obj):
                return asdict(obj)
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, (list, tuple)):
                return [serialize(item) for item in obj]
            elif isinstance(obj, float):
                return round(obj, 6)
            return obj

        serialized = serialize({"args": args, "kwargs": kwargs})
        data = str(serialized).encode("utf-8")
        return hashlib.md5(data, usedforsecurity=False).hexdigest()

    def get_or_compute(
        self,
        compute_fn: Callable[[], Any],
        *args,
        **kwargs,
    ) -> Any:
        """Retrieve from cache or compute and store."""
        key = self._hash_inputs(*args, **kwargs)
        current_time = time.time()

        if key in self._cache:
            entry = self._cache[key]
            if current_time - entry.timestamp < self._ttl:
                entry.hits += 1
                self._hits += 1
                self._cache.move_to_end(key)
                return entry.result
            else:
                del self._cache[key]

        self._misses += 1
        result = compute_fn()

        self._cache[key] = CacheEntry(
            result=result,
            timestamp=current_time,
            input_hash=key,
        )

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return result

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self._ttl,
        }


# Global cache instance
_analysis_cache = AnalysisCache()


def get_cache_stats() -> dict[str, Any]:
    """Return statistics for the global analysis cache."""
    return _analysis_cache.stats()


def clear_cache() -> None:
    """Clear the global analysis cache."""
    _analysis_cache.clear()


def memoize_analysis(func: Callable) -> Callable:
    """Decorator to memoize analysis functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return _analysis_cache.get_or_compute(
            lambda: func(*args, **kwargs),
            func.__name__,
            *args,
            **kwargs,
        )
    return wrapper
