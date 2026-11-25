"""Pricing, token counting, caches, backbones config, and embedders."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from guide_exemplars.core.protocols import FREE, LLMRequest, ModelPrice
from guide_exemplars.llm.backbones import build_embedder, build_llm, interpolate, load_backbones
from guide_exemplars.llm.backends import OpenAIChatBackend
from guide_exemplars.llm.cache import (
    MemoryEmbeddingStore,
    SQLiteEmbeddingStore,
    SQLiteResponseCache,
    text_sha,
)
from guide_exemplars.llm.embedders import CachedEmbedder, HashEmbedder
from guide_exemplars.llm.pricing import PriceTable
from guide_exemplars.llm.tokens import TokenCounter

# ---- pricing ----------------------------------------------------------------------------------------


def test_model_price_cost_with_cached_tokens() -> None:
    p = ModelPrice(input=0.15, cached_input=0.075, output=0.60)
    # 1000 prompt (400 cached) + 100 completion
    assert p.cost(1000, 400, 100) == pytest.approx((600 * 0.15 + 400 * 0.075 + 100 * 0.60) / 1e6)
    assert FREE.cost(10**6, 0, 10**6) == 0.0


def test_price_table_lookup_with_and_without_provider_prefix() -> None:
    table = PriceTable.load()
    assert table.lookup("gpt-4o-mini") == table.lookup("openai/gpt-4o-mini")
    assert table.lookup("openai/gpt-4o-mini").output == pytest.approx(0.60)
    assert table.lookup("unknown/model") is None


# ---- tokens -----------------------------------------------------------------------------------------


def test_token_counter() -> None:
    tc = TokenCounter()
    assert tc.count("") == 0
    assert 1 <= tc.count("hello world") <= 4
    assert tc.count_messages((("system", "a"), ("user", "b"))) > tc.count("ab")
    fallback = TokenCounter("no-such-encoding")
    assert not fallback.exact and fallback.count("x" * 40) == 10


# ---- caches -----------------------------------------------------------------------------------------


def test_sqlite_response_cache_persists(tmp_path: Path) -> None:
    db = tmp_path / "cache.db"
    c1 = SQLiteResponseCache(db)
    c1.put("k", "m", {"text": "hi", "n": 1})
    c1.close()
    c2 = SQLiteResponseCache(db)
    assert c2.get("k") == {"text": "hi", "n": 1}
    assert c2.get("missing") is None
    assert len(c2) == 1


def test_sqlite_embedding_store_roundtrip(tmp_path: Path) -> None:
    store = SQLiteEmbeddingStore(tmp_path / "cache.db")
    v = np.arange(4, dtype=np.float32)
    store.put_many("m", {"a": v})
    got = store.get_many("m", ["a", "b"])
    assert set(got) == {"a"}
    np.testing.assert_array_equal(got["a"], v)
    assert store.get_many("other-model", ["a"]) == {}


# ---- embedders --------------------------------------------------------------------------------------


async def test_hash_embedder_deterministic_normalized_and_semantic() -> None:
    e = HashEmbedder(dim=128)
    vecs = await e.embed(["the cat sat on the mat", "the cat sat on a mat", "quantum flux capacitor"])
    again = await HashEmbedder(dim=128).embed(["the cat sat on the mat"])
    np.testing.assert_allclose(vecs[0], again[0])
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]


async def test_cached_embedder_only_embeds_new_texts() -> None:
    calls: list[list[str]] = []

    class Counting(HashEmbedder):
        async def embed(self, texts):
            calls.append(list(texts))
            return await super().embed(texts)

    store = MemoryEmbeddingStore()
    emb = CachedEmbedder(Counting(64), store)
    first = await emb.embed(["a b", "c d", "a b"])
    second = await emb.embed(["c d", "e f"])
    assert calls == [["a b", "c d"], ["e f"]]
    np.testing.assert_allclose(first[1], second[0])
    assert set(store.get_many(emb.model_id, [text_sha("a b"), text_sha("e f")])) == {
        text_sha("a b"),
        text_sha("e f"),
    }


def test_build_embedder_specs() -> None:
    assert build_embedder("hash").model_id == "hash-256"
    assert build_embedder("hash-64").model_id == "hash-64"
    assert build_embedder("st/all-MiniLM-L6-v2").model_id == "st/all-MiniLM-L6-v2"
    assert build_embedder("openai/text-embedding-3-small", api_key="x").model_id.endswith("3-small")
    with pytest.raises(ValueError):
        build_embedder("bogus")


# ---- backbones --------------------------------------------------------------------------------------


def test_interpolation() -> None:
    env = {"A": "1", "EMPTY": ""}
    assert interpolate("x${A}y", env) == "x1y"
    assert interpolate("${MISSING:-dflt}", env) == "dflt"
    assert interpolate("${EMPTY:-dflt}", env) == "dflt"
    assert interpolate("${MISSING}", env) == ""
    assert interpolate({"k": ["${A}", 3]}, env) == {"k": ["1", 3]}


def test_shipped_backbones_load_and_report_readiness() -> None:
    env = {"QWEN_API_BASE": "http://box:8000/v1"}
    bb = load_backbones(env=env)
    assert {"gpt4o-mini", "qwen", "fake"} <= set(bb)
    assert bb["qwen"].api_base == "http://box:8000/v1"
    assert bb["qwen"].model == "orcarouter/Qwen3.8-27B-Uncensored-FP8"
    assert bb["fake"].missing_settings() == []
    no_base = load_backbones(env={})["qwen"]
    assert no_base.api_base is None
    assert any("api_base" in m for m in no_base.missing_settings())


def test_redacted_never_contains_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    cfg = load_backbones()["gpt4o-mini"]
    assert cfg.api_key() == "sk-secret-value"
    assert "sk-secret-value" not in str(cfg.redacted())


async def test_build_llm_fake_and_self_hosted_pricing() -> None:
    bb = load_backbones(env={"QWEN_API_BASE": "http://box:8000/v1"})
    fake = build_llm(bb["fake"])
    res = await fake.complete(LLMRequest.of("s", "u", 10, "check"))
    assert res.text == '{"ok": true}' and res.cost_usd > 0
    assert build_llm(bb["qwen"]).price == FREE
    assert build_llm(bb["gpt4o-mini"]).price.input == pytest.approx(0.15)


# ---- OpenAI backend ---------------------------------------------------------------------------------


async def test_openai_backend_request_and_usage_parsing() -> None:
    sent: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            sent.update(kwargs)
            usage = SimpleNamespace(
                prompt_tokens=1200,
                completion_tokens=40,
                prompt_tokens_details=SimpleNamespace(cached_tokens=1024),
            )
            msg = SimpleNamespace(content='{"label": "2"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    backend = OpenAIChatBackend(
        model="openai/gpt-4o-mini", extra_body={"x": 1}, max_tokens_param="max_completion_tokens"
    )
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    resp = await backend.complete(LLMRequest.of("s", "u", 64, "grade"), temperature=0.2)
    assert (resp.text, resp.prompt_tokens, resp.cached_prompt_tokens, resp.completion_tokens) == (
        '{"label": "2"}',
        1200,
        1024,
        40,
    )
    assert sent["model"] == "gpt-4o-mini"
    assert sent["max_completion_tokens"] == 64 and "max_tokens" not in sent
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["extra_body"] == {"x": 1}
    assert sent["messages"][0] == {"role": "system", "content": "s"}


async def test_openai_backend_tolerates_missing_usage() -> None:
    class Bare:
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[], usage=None)

    backend = OpenAIChatBackend(model="m", json_mode_supported=False)
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=Bare()))
    resp = await backend.complete(LLMRequest.of("s", "u", 8, "grade"), 0.0)
    assert (resp.text, resp.prompt_tokens) == ("", 0)
    assert "response_format" not in backend.request_kwargs(LLMRequest.of("s", "u", 8, "grade"), 0.0)


# ---- fatal provider errors --------------------------------------------------------------------------


def test_account_level_errors_become_provider_fatal() -> None:
    from guide_exemplars.core.protocols import AbortRun, ProviderFatal
    from guide_exemplars.llm.backends import raise_if_fatal

    class RateLimitError(Exception):
        def __init__(self, msg: str, code: str | None = None):
            super().__init__(msg)
            self.code = code

    class AuthenticationError(Exception):
        pass

    with pytest.raises(ProviderFatal):
        raise_if_fatal(RateLimitError("You exceeded your current quota", code="insufficient_quota"))
    with pytest.raises(ProviderFatal):
        raise_if_fatal(AuthenticationError("bad key"))
    raise_if_fatal(RateLimitError("Rate limit reached for requests"))  # transient: left for SDK retries
    assert issubclass(ProviderFatal, AbortRun)


async def test_openai_backend_quota_error_aborts() -> None:
    from guide_exemplars.core.protocols import ProviderFatal

    class Quota:
        async def create(self, **kwargs):
            err = Exception("Error code: 429 - insufficient_quota")
            raise err

    backend = OpenAIChatBackend(model="m")
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=Quota()))
    with pytest.raises(ProviderFatal):
        await backend.complete(LLMRequest.of("s", "u", 8, "grade"), 0.0)


# ---- token-rate limiter ------------------------------------------------------------------------------------


def test_rate_limiter_paces_in_arrival_order() -> None:
    from guide_exemplars.llm.ratelimit import TokenRateLimiter

    now = [0.0]
    lim = TokenRateLimiter(600, clock=lambda: now[0])  # 10 tokens/s, bucket of 600
    assert lim.reserve(600) == 0.0  # a full bucket goes out at once
    assert lim.reserve(100) == pytest.approx(10.0)  # then each call waits for its share
    assert lim.reserve(100) == pytest.approx(20.0)
    now[0] = 30.0  # time passes: the debt is paid off and 100 tokens have refilled
    assert lim.reserve(100) == 0.0
    assert lim.reserve(5000) == pytest.approx(60.0)  # an oversized call waits for a full bucket, not forever


def test_shared_rate_limiter_is_per_model_and_skips_fake() -> None:
    from guide_exemplars.llm.backbones import BackboneConfig, shared_rate_limiter

    a = BackboneConfig(name="a", model="openai/gpt-4o-mini", tpm_limit=1000)
    b = BackboneConfig(name="b", model="openai/gpt-4o-mini", tpm_limit=1000)
    assert shared_rate_limiter(a) is shared_rate_limiter(b)
    assert shared_rate_limiter(BackboneConfig(name="c", model="openai/gpt-4o-mini")) is None
    assert shared_rate_limiter(BackboneConfig(name="f", kind="fake", model="fake/grader", tpm_limit=10)) is None
