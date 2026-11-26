from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from guide_exemplars.core.feature_grader import (
    FeatureExtractor,
    OrdinalRidge,
    apply_thresholds,
    fit_thresholds,
)
from guide_exemplars.core.metrics import quadratic_weighted_kappa
from guide_exemplars.core.prompts import PromptBuilder, parse_counts
from guide_exemplars.core.protocols import FREE, LLMRequest, ProviderUnstable
from guide_exemplars.core.task import TaskSpec
from guide_exemplars.core.types import Item
from guide_exemplars.llm.cache import MemoryResponseCache
from guide_exemplars.llm.client import LLMClient
from guide_exemplars.llm.fake import FakeBackend
from support.toy import simulator_llm, toy_items, toy_task


def test_parse_counts_requires_every_name_as_a_whole_non_negative_number() -> None:
    assert parse_counts('{"a": 3, "b": "2"}', ["a", "b"]) == {"a": 3, "b": 2}
    assert parse_counts('noise {"a": 0, "b": 4.0} trailing', ["a", "b"]) == {"a": 0, "b": 4}
    for bad in ('{"a": 3}', '{"a": -1, "b": 2}', '{"a": 1.5, "b": 2}', '{"a": "x", "b": 2}', "no json"):
        assert parse_counts(bad, ["a", "b"]) is None


def test_features_prompt_lists_counts_and_json_skeleton() -> None:
    prompts = PromptBuilder(toy_task())
    system = prompts.features_system
    assert "- gaps: claims made without support" in system
    assert '{"gaps": <count>, "vague": <count>}' in system
    assert "<response_to_score>\nhello world\n</response_to_score>" in prompts.features_user("hello world")


def test_task_without_features_has_no_feature_prompt_and_keeps_its_hash() -> None:
    data = toy_task().model_dump(mode="json")
    data.pop("features")
    spec = TaskSpec.model_validate(data)
    with pytest.raises(ValueError, match="defines no features"):
        _ = PromptBuilder(spec).features_system
    # The hash of a spec without features is the hash of its dump without the field (as before it existed).
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
    assert spec.spec_hash() == hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    assert toy_task().spec_hash() != spec.spec_hash()


def test_feature_spec_rejects_duplicate_names() -> None:
    data = toy_task().model_dump(mode="json")
    data["features"]["counts"].append({"name": "gaps", "description": "again"})
    with pytest.raises(ValueError, match="duplicate"):
        TaskSpec.model_validate(data)


def test_fit_thresholds_recovers_a_separable_ordering() -> None:
    y = np.repeat([0, 1, 2, 3], 25)
    scores = y + np.random.default_rng(0).uniform(-0.3, 0.3, len(y))
    t = fit_thresholds(scores, y, 4)
    assert np.all(np.diff(t) >= 0)
    assert np.array_equal(apply_thresholds(scores, t), y)


def test_fit_thresholds_beats_proportional_cuts_on_a_compressed_score() -> None:
    # A grader whose scores for the top level overlap the level below: proportional cuts are the start point,
    # coordinate ascent must not end below them.
    rng = np.random.default_rng(1)
    y = rng.choice(4, 400, p=[0.05, 0.2, 0.55, 0.2])
    scores = np.minimum(y, 2) + 0.4 * (y == 3) + rng.normal(0, 0.35, len(y))
    props = np.cumsum(np.bincount(y, minlength=4))[:-1] / len(y)
    start = quadratic_weighted_kappa(
        y.tolist(), apply_thresholds(scores, np.quantile(scores, props)).tolist(), 4
    )
    fitted = quadratic_weighted_kappa(
        y.tolist(), apply_thresholds(scores, fit_thresholds(scores, y, 4)).tolist(), 4
    )
    assert fitted >= start


def test_ordinal_ridge_learns_a_signal_and_imputes_missing_values() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, 300)
    X = np.column_stack([-y + rng.normal(0, 0.4, len(y)), rng.normal(0, 1, len(y))])
    model = OrdinalRidge(4).fit(X, y)
    assert model.coef_[0] < 0 and abs(model.coef_[0]) > 5 * abs(model.coef_[1])
    assert model.fit_qwk_ > 0.8
    X_new = X.copy()
    X_new[:10, 0] = math.nan
    pred = model.predict(X_new)
    assert pred.shape == (300,) and set(pred.tolist()) <= {0, 1, 2, 3}
    desc = model.describe(["signal", "noise"])
    assert set(desc["coef_standardized"]) == {"signal", "noise"} and len(desc["thresholds"]) == 3


async def test_extractor_counts_with_the_simulator_and_vectorizes_rates() -> None:
    items = toy_items(n_per_label=6)
    extractor = FeatureExtractor(PromptBuilder(toy_task()), simulator_llm(items))
    assert extractor.feature_names == ["gaps_per_100w", "vague_per_100w", "log_words"]
    extractions = await extractor.extract(items)
    assert all(ex.counts is not None for ex in extractions)
    X = extractor.vectorize(items, extractions)
    assert X.shape == (len(items), 3)
    words = len(items[0].text.split())
    assert X[0, 0] == pytest.approx(extractions[0].counts["gaps"] / words * 100)
    assert X[0, 2] == pytest.approx(math.log1p(words))
    # The simulator's counts fall with the level, so the fitted model separates levels on the training data.
    y = [toy_task().scale.rank(it.label) for it in items]
    assert OrdinalRidge(4).fit(X, y).fit_qwk_ > 0.5


async def test_extractor_requests_are_deterministic_and_cached() -> None:
    items = toy_items(n_per_label=2)
    cache = MemoryResponseCache()
    llm = simulator_llm(items, cache=cache)
    extractor = FeatureExtractor(PromptBuilder(toy_task()), llm)
    req = extractor.request(items[0].text)
    assert req.purpose == "features" and req.temperature == 0.0
    first = await extractor.extract(items)
    second = await extractor.extract(items)
    assert [e.counts for e in first] == [e.counts for e in second]
    assert len(cache) == len(items)


def _client(responder) -> LLMClient:
    return LLMClient(FakeBackend(responder=responder), "fake/grader", price=FREE, cache=MemoryResponseCache())


async def test_unparsable_replies_are_nan_rows_not_failures() -> None:
    items = [Item(item_id=f"i{i}", text=f"text {i}", label="Beginning") for i in range(5)]
    extractor = FeatureExtractor(PromptBuilder(toy_task()), _client(lambda _req: "not json"))
    extractions = await extractor.extract(items)
    assert all(ex.counts is None and ex.error for ex in extractions)
    X = extractor.vectorize(items, extractions)
    assert np.isnan(X[:, :2]).all() and not np.isnan(X[:, 2]).any()


async def test_burst_of_provider_errors_aborts_extraction() -> None:
    def boom(_req: LLMRequest) -> str:
        raise RuntimeError("provider down")

    items = [Item(item_id=f"i{i}", text=f"text {i}", label="Beginning") for i in range(10)]
    extractor = FeatureExtractor(PromptBuilder(toy_task()), _client(boom))
    with pytest.raises(ProviderUnstable):
        await extractor.extract(items)
