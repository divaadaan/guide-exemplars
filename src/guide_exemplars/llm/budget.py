"""Spend accounting: a reservation-based hard cap and a usage meter.

Before each call the client reserves a worst-case cost: prompt tokens at the uncached rate plus ``max_tokens``
of output. After the call it commits the actual cost. Concurrent calls therefore can never push spend past the
cap, however many are in flight.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from guide_exemplars.core.protocols import BudgetExceeded


class Budget:
    def __init__(self, max_usd: float | None, spent_usd: float = 0.0):
        if max_usd is not None and max_usd < 0:
            raise ValueError("max_usd must be non-negative")
        self.max_usd = max_usd
        self._spent = float(spent_usd)
        self._reserved = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def reserved(self) -> float:
        return self._reserved

    @property
    def remaining(self) -> float | None:
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self._spent - self._reserved)

    def reserve(self, amount: float) -> float:
        amount = max(0.0, float(amount))
        with self._lock:
            if self.max_usd is not None and self._spent + self._reserved + amount > self.max_usd:
                raise BudgetExceeded(
                    f"call needs up to ${amount:.4f}; spent ${self._spent:.4f}, reserved "
                    f"${self._reserved:.4f}, cap ${self.max_usd:.4f}"
                )
            self._reserved += amount
        return amount

    def commit(self, reservation: float, actual: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)
            self._spent += max(0.0, float(actual))

    def release(self, reservation: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)


@dataclass(frozen=True)
class UsageRecord:
    purpose: str
    model: str
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cache_hit: bool
    latency_ms: float
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UsageTotals:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, r: UsageRecord) -> None:
        self.calls += 1
        self.cache_hits += int(r.cache_hit)
        if not r.cache_hit:
            self.prompt_tokens += r.prompt_tokens
            self.cached_prompt_tokens += r.cached_prompt_tokens
            self.completion_tokens += r.completion_tokens
        self.cost_usd += r.cost_usd

    def to_dict(self) -> dict:
        return asdict(self)


class UsageMeter:
    """Thread-safe usage log with running totals, overall and per purpose."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[UsageRecord] = []
        self.total = UsageTotals()
        self.by_purpose: dict[str, UsageTotals] = {}
        self._listeners: list[Callable[[UsageRecord], None]] = []

    def subscribe(self, fn: Callable[[UsageRecord], None]) -> None:
        self._listeners.append(fn)

    def record(self, r: UsageRecord) -> None:
        with self._lock:
            self.records.append(r)
            self.total.add(r)
            self.by_purpose.setdefault(r.purpose, UsageTotals()).add(r)
            listeners = list(self._listeners)
        for fn in listeners:
            fn(r)
