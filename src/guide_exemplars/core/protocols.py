"""Ports between the core algorithm and the outside world: LLMs, embedders, events, cancellation.

The core depends only on these protocols. Provider SDKs, caches, budgets and the service live in other packages
and implement them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

Purpose = Literal["grade", "rationale_plain", "rationale_disc", "features", "serve", "check", "embed"]


class AbortRun(RuntimeError):
    """Base for conditions that must stop a whole evaluation, never be recorded as one item's failure."""


class BudgetExceeded(AbortRun):
    """Raised before a call that could push spend past the hard cap."""


class Cancelled(AbortRun):
    """Raised when a cancel token is set mid-run."""


class CacheMiss(AbortRun):
    """Raised in replay (cache-only) mode when a request has no cached response."""


class ProviderFatal(AbortRun):
    """Account-level provider failure (exhausted credits/quota, bad key, unknown model). Retrying item by item
    cannot help. Recording it as a per-item grading failure would silently corrupt results, so the run stops and
    can be resumed from its checkpoint once the account is fixed."""


class ProviderUnstable(AbortRun):
    """Too many provider errors (rate limits, timeouts, 5xx) in one evaluation, even after retries. A few failed
    calls count as wrong answers, as in the paper, but a burst means the provider, not the exemplar set, decided
    the score. The run stops and can be resumed from its checkpoint."""


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens."""

    input: float
    cached_input: float
    output: float

    def cost(self, prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int) -> float:
        uncached = max(0, prompt_tokens - cached_prompt_tokens)
        return (
            uncached * self.input + cached_prompt_tokens * self.cached_input + completion_tokens * self.output
        ) / 1_000_000


FREE = ModelPrice(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[tuple[str, str], ...]  # (role, content)
    max_tokens: int
    purpose: Purpose
    temperature: float | None = None  # None → the backbone's default
    json_mode: bool = True

    @classmethod
    def of(cls, system: str, user: str, max_tokens: int, purpose: Purpose, **kw: object) -> LLMRequest:
        return cls(
            messages=(("system", system), ("user", user)), max_tokens=max_tokens, purpose=purpose, **kw
        )

    def as_openai_messages(self) -> list[dict[str, str]]:
        return [{"role": role, "content": content} for role, content in self.messages]


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cache_hit: bool
    latency_ms: float = 0.0


class LLM(Protocol):
    model_id: str

    async def complete(self, req: LLMRequest) -> LLMResult: ...

    def count_tokens(self, text: str) -> int: ...

    @property
    def price(self) -> ModelPrice: ...


class Embedder(Protocol):
    model_id: str

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """(n, d) float array, one row per text."""
        ...


class CancelToken(Protocol):
    def is_set(self) -> bool: ...


class NeverCancel:
    def is_set(self) -> bool:
        return False
