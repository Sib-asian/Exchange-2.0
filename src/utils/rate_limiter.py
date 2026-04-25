"""
rate_limiter.py — Sliding-window rate limiter for external API calls.

Prevents excessive API consumption (Gemini, OpenAI, OpenWeather) by
throttling calls when the rate limit is approached.

Usage:
    from src.utils.rate_limiter import RateLimiter
    
    _limiter = RateLimiter(max_calls=60, window_seconds=60)
    
    async def call_api():
        _limiter.wait_if_needed()
        # make API call...
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

_LOG = logging.getLogger("exchange.rate_limiter")


class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, max_calls: int = 60, window_seconds: float = 60.0) -> None:
        """
        Args:
            max_calls: Maximum number of calls allowed within the window.
            window_seconds: Duration of the sliding window in seconds.
        """
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """
        Block until a call is allowed under the rate limit.
        
        If the current call count within the window exceeds max_calls,
        sleeps until the oldest call expires from the window.
        """
        with self._lock:
            now = time.monotonic()
            # Remove expired timestamps
            while self._timestamps and self._timestamps[0] <= now - self._window:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._max_calls:
                # Calculate sleep time: wait until oldest timestamp exits window
                sleep_time = self._timestamps[0] - (now - self._window) + 0.01
                if sleep_time > 0:
                    _LOG.info(
                        "Rate limit reached (%d/%d calls in %.0fs). Sleeping %.1fs.",
                        len(self._timestamps), self._max_calls, self._window, sleep_time,
                    )
                    time.sleep(sleep_time)

            self._timestamps.append(time.monotonic())

    @property
    def current_usage(self) -> int:
        """Return the number of calls in the current window."""
        with self._lock:
            now = time.monotonic()
            while self._timestamps and self._timestamps[0] <= now - self._window:
                self._timestamps.popleft()
            return len(self._timestamps)

    def reset(self) -> None:
        """Clear all tracked timestamps."""
        with self._lock:
            self._timestamps.clear()


# Pre-configured limiters for common APIs
GEMINI_LIMITER = RateLimiter(max_calls=50, window_seconds=60)    # Gemini free tier: 1500/day ≈ ~62/min safe
OPENAI_LIMITER = RateLimiter(max_calls=30, window_seconds=60)     # Conservative default
OPENWEATHER_LIMITER = RateLimiter(max_calls=50, window_seconds=60)  # OpenWeather: 60/min free tier
