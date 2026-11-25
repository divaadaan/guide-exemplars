from __future__ import annotations

import asyncio

import pytest

from guide_exemplars.core.protocols import BudgetExceeded, LLMRequest, ModelPrice
from guide_exemplars.llm.budget import Budget, UsageMeter
from guide_exemplars.llm.client import LLMClient, strip_think
from guide_exemplars.llm.fake import FakeBackend, FakeBackendError, constant_responder

PRICE = ModelPrice(input=1.0, cached_input=0.5, output=2.0)  # $ per 1M tokens


def req(user: str = "hello", **kw) -> LLMRequest:
    return LLMRequest.of("system prompt", user, max_tokens=kw.pop("max_tokens", 50), purpose="grade", **kw)


def client(backend: FakeBackend | None = None, **kw) -> LLMClient:
    return LLMClient(backend or FakeBackend(), "fake/grader", price=kw.pop("price", PRICE), **kw)


async def test_cache_hit_is_free_and_skips_backend() -> None:
    backend = FakeBackend()
    llm = client(backend)
    first = await llm.complete(req())
    second = await llm.complete(req())
    assert backend.calls == 1
    assert not first.cache_hit and first.cost_usd > 0
    assert second.cache_hit and second.cost_usd == 0.0
    assert second.text == first.text
    assert llm.meter.total.calls == 2 and llm.meter.total.cache_hits == 1


async def test_cache_key_sensitive_to_every_output_affecting_field() -> None:
    llm = client()
    base = llm.cache_key(req(), 0.2)
    assert llm.cache_key(req(), 0.3) != base
    assert llm.cache_key(req(max_tokens=51), 0.2) != base
    assert llm.cache_key(req(user="other"), 0.2) != base
    assert llm.cache_key(req(json_mode=False), 0.2) != base
    other_model = LLMClient(FakeBackend(), "fake/other", price=PRICE)
    assert other_model.cache_key(req(), 0.2) != base
    other_ns = client(cache_namespace="http://other:8000/v1")
    assert other_ns.cache_key(req(), 0.2) != base


async def test_request_temperature_overrides_default() -> None:
    llm = client(default_temperature=0.2)
    await llm.complete(req())
    await llm.complete(req(temperature=0.9))
    assert llm.meter.total.cache_hits == 0


async def test_budget_never_overshoots_under_concurrency() -> None:
    budget = Budget(max_usd=0.0015)
    llm = client(FakeBackend(jitter_s=0.01), budget=budget, max_concurrency=16)
    results = await asyncio.gather(
        *(llm.complete(req(user=f"item {i}")) for i in range(64)), return_exceptions=True
    )
    ok = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, BudgetExceeded)]
    assert ok and refused
    assert len(ok) + len(refused) == 64
    assert budget.spent <= budget.max_usd
    assert budget.reserved == pytest.approx(0.0)
    assert budget.spent == pytest.approx(sum(r.cost_usd for r in ok))


async def test_failed_call_releases_reservation_and_is_not_cached() -> None:
    backend = FakeBackend(failure_rate=1.0)
    budget = Budget(max_usd=1.0)
    llm = client(backend, budget=budget)
    with pytest.raises(FakeBackendError):
        await llm.complete(req())
    assert budget.reserved == 0.0 and budget.spent == 0.0
    with pytest.raises(FakeBackendError):
        await llm.complete(req())
    assert backend.calls == 2


def test_unpriced_model_with_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="no price"):
        LLMClient(FakeBackend(), "mystery", price=None, budget=Budget(1.0))
    LLMClient(FakeBackend(), "mystery", price=None, budget=Budget(1.0), allow_unpriced=True)
    LLMClient(FakeBackend(), "mystery", price=None)  # no cap: fine, priced at zero


async def test_strip_reasoning() -> None:
    backend = FakeBackend(responder=constant_responder('<think>hmm\nmaybe 2</think>\n{"label": "3"}'))
    llm = client(backend, strip_reasoning=True)
    assert (await llm.complete(req())).text == '{"label": "3"}'
    assert strip_think('reasoning only</think>{"a": 1}') == '{"a": 1}'
    assert strip_think('{"a": 1}') == '{"a": 1}'


async def test_meter_tracks_purposes_and_listeners() -> None:
    meter = UsageMeter()
    seen = []
    meter.subscribe(seen.append)
    llm = client(meter=meter)
    await llm.complete(req())
    await llm.complete(LLMRequest.of("s", "u", 10, "rationale_plain"))
    assert set(meter.by_purpose) == {"grade", "rationale_plain"}
    assert len(seen) == 2


def test_budget_validation_and_remaining() -> None:
    with pytest.raises(ValueError):
        Budget(-1)
    b = Budget(1.0, spent_usd=0.25)
    r = b.reserve(0.5)
    assert b.remaining == pytest.approx(0.25)
    b.commit(r, 0.1)
    assert b.spent == pytest.approx(0.35) and b.reserved == 0
    with pytest.raises(BudgetExceeded):
        b.reserve(0.7)
    assert Budget(None).remaining is None
