"""LLMClient: the one wrapper every backend goes through.

Per call: check the response cache, reserve the worst-case cost against the budget, wait for token-rate headroom
(when a TPM limit is set), wait for a concurrency slot, call the backend, price the real usage, commit the spend,
meter it, and cache the result. Failed calls release their reservation and are never cached.
"""

from __future__ import annotations

import asyncio
import re
import time

from guide_exemplars.core.protocols import FREE, CacheMiss, LLMRequest, LLMResult, ModelPrice
from guide_exemplars.llm.backends import ChatBackend
from guide_exemplars.llm.budget import Budget, UsageMeter, UsageRecord
from guide_exemplars.llm.cache import MemoryResponseCache, ResponseCache, response_cache_key
from guide_exemplars.llm.ratelimit import TokenRateLimiter
from guide_exemplars.llm.tokens import TokenCounter

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove reasoning traces some open models emit before the answer."""
    text = _THINK.sub("", text)
    if "</think>" in text.lower():  # unterminated opening tag: keep what follows the last close tag
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


class LLMClient:
    def __init__(
        self,
        backend: ChatBackend,
        model_id: str,
        *,
        price: ModelPrice | None,
        cache: ResponseCache | None = None,
        meter: UsageMeter | None = None,
        budget: Budget | None = None,
        max_concurrency: int = 16,
        default_temperature: float = 0.2,
        strip_reasoning: bool = False,
        cache_namespace: str = "",
        extra_cache_key: object = None,
        tokens: TokenCounter | None = None,
        allow_unpriced: bool = False,
        cache_only: bool = False,
        rate_limiter: TokenRateLimiter | None = None,
    ):
        if price is None and budget is not None and budget.max_usd is not None and not allow_unpriced:
            raise ValueError(
                f"no price known for {model_id!r}; add it to configs/pricing.yaml, mark the backbone "
                "self_hosted, or pass allow_unpriced=True"
            )
        self.backend = backend
        self.model_id = model_id
        self._price = price if price is not None else FREE
        self.cache = cache if cache is not None else MemoryResponseCache()
        self.meter = meter if meter is not None else UsageMeter()
        self.budget = budget
        self.max_concurrency = max(1, int(max_concurrency))
        self.default_temperature = default_temperature
        self.strip_reasoning = strip_reasoning
        self.cache_namespace = cache_namespace
        self.extra_cache_key = extra_cache_key
        self.tokens = tokens or TokenCounter()
        self.cache_only = cache_only  # replay mode: any cache miss aborts the run
        self.rate_limiter = rate_limiter
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    @property
    def price(self) -> ModelPrice:
        return self._price

    def count_tokens(self, text: str) -> int:
        return self.tokens.count(text)

    def _semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        sem = self._semaphores.get(loop_id)
        if sem is None:
            sem = self._semaphores[loop_id] = asyncio.Semaphore(self.max_concurrency)
        return sem

    def cache_key(self, req: LLMRequest, temperature: float) -> str:
        return response_cache_key(
            model=self.model_id,
            namespace=self.cache_namespace,
            temperature=temperature,
            max_tokens=req.max_tokens,
            json_mode=req.json_mode,
            extra=self.extra_cache_key,
            messages=req.messages,
        )

    def worst_case_cost(self, req: LLMRequest) -> float:
        return self._price.cost(self.tokens.count_messages(req.messages), 0, req.max_tokens)

    async def complete(self, req: LLMRequest) -> LLMResult:
        temperature = self.default_temperature if req.temperature is None else req.temperature
        key = self.cache_key(req, temperature)

        hit = self.cache.get(key)
        if hit is not None:
            result = LLMResult(
                text=hit["text"],
                prompt_tokens=hit["prompt_tokens"],
                cached_prompt_tokens=hit["cached_prompt_tokens"],
                completion_tokens=hit["completion_tokens"],
                cost_usd=0.0,
                cache_hit=True,
            )
            self._record(req, result)
            return result

        if self.cache_only:
            raise CacheMiss(f"no cached response for a {req.purpose} request ({key[:12]})")
        reservation = self.budget.reserve(self.worst_case_cost(req)) if self.budget else 0.0
        started = time.perf_counter()
        try:
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire(self.tokens.count_messages(req.messages) + req.max_tokens)
            async with self._semaphore():
                resp = await self.backend.complete(req, temperature)
        except BaseException:
            if self.budget:
                self.budget.release(reservation)
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        cost = self._price.cost(resp.prompt_tokens, resp.cached_prompt_tokens, resp.completion_tokens)
        if self.budget:
            self.budget.commit(reservation, cost)

        text = strip_think(resp.text) if self.strip_reasoning else resp.text
        result = LLMResult(
            text=text,
            prompt_tokens=resp.prompt_tokens,
            cached_prompt_tokens=resp.cached_prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=cost,
            cache_hit=False,
            latency_ms=latency_ms,
        )
        self.cache.put(
            key,
            self.model_id,
            {
                "text": text,
                "prompt_tokens": resp.prompt_tokens,
                "cached_prompt_tokens": resp.cached_prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            },
        )
        self._record(req, result)
        return result

    def _record(self, req: LLMRequest, r: LLMResult) -> None:
        self.meter.record(
            UsageRecord(
                purpose=req.purpose,
                model=self.model_id,
                prompt_tokens=r.prompt_tokens,
                cached_prompt_tokens=r.cached_prompt_tokens,
                completion_tokens=r.completion_tokens,
                cost_usd=r.cost_usd,
                cache_hit=r.cache_hit,
                latency_ms=r.latency_ms,
            )
        )
