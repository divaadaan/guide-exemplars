"""Client-side token-rate limiting, so a run stays under the provider's tokens-per-minute (TPM) limit.

Providers meter TPM on prompt tokens plus the requested ``max_tokens``, and they count prefix-cached tokens too.
Bursting past the limit makes calls fail after the SDK's retries run out, and a failed grading call counts as a
wrong answer, so pacing here protects the results, not just the provider.

The limiter is a token bucket with debt: each call takes its tokens immediately (the level may go negative) and
then sleeps until the bucket would have refilled. Calls are paced in arrival order, and it is thread-safe because
the service may run several event loops.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable


class TokenRateLimiter:
    def __init__(self, tokens_per_minute: float, clock: Callable[[], float] = time.monotonic):
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        self.capacity = float(tokens_per_minute)
        self.rate = self.capacity / 60.0  # tokens per second
        self._clock = clock
        self._level = self.capacity
        self._last = clock()
        self._lock = threading.Lock()

    def reserve(self, tokens: int) -> float:
        """Take ``tokens`` now and return how many seconds the caller must wait before sending."""
        need = min(float(tokens), self.capacity)  # a single oversized call waits for a full bucket, never forever
        with self._lock:
            now = self._clock()
            self._level = min(self.capacity, self._level + (now - self._last) * self.rate)
            self._last = now
            self._level -= need
            return max(0.0, -self._level / self.rate)

    async def acquire(self, tokens: int) -> float:
        wait = self.reserve(tokens)
        if wait > 0:
            await asyncio.sleep(wait)
        return wait
