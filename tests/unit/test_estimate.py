from __future__ import annotations

import pytest

from guide_exemplars.core.estimate import estimate_run
from guide_exemplars.core.optimizer import GuideConfig
from guide_exemplars.core.protocols import ModelPrice
from guide_exemplars.llm.tokens import TokenCounter
from support.toy import toy_items, toy_split, toy_task

PRICE = ModelPrice(0.15, 0.075, 0.60)
COUNT = TokenCounter().count
TASK = toy_task()
TRAIN, VAL, EXPERTS = toy_split(toy_items())


def est(cfg: GuideConfig | None = None, val=VAL, **kw):
    return estimate_run(TASK, TRAIN, val, cfg or GuideConfig(), COUNT, PRICE, n_experts=len(EXPERTS), **kw)


def test_grading_call_count_matches_optimizer_schedule() -> None:
    e = est(GuideConfig(n_rounds=5, n_eval=32))
    V = len(VAL)
    assert e.groups["grade"].calls == 5 * 32 * V - 4 * V
    assert e.groups["rationale_plain"].calls == len(TRAIN) + len(EXPERTS)
    assert e.groups["rationale_disc"].calls == 4 * len(TRAIN)
    assert "rationale_disc" not in est(GuideConfig(phase2=False)).groups


def test_price_ordering() -> None:
    e = est()
    assert 0 < e.usd_low < e.usd_high < e.usd_hard_max


def test_monotonic_in_validation_rounds_and_set_size() -> None:
    base = est().usd_high
    assert est(val=VAL + VAL).usd_high > base
    assert est(GuideConfig(n_rounds=6)).usd_high > base
    assert est(GuideConfig(k_min=8, k_max=16)).usd_high > est(GuideConfig(k_min=4, k_max=8)).usd_high


def test_test_scoring_only() -> None:
    e = est(include_optimization=False, test=VAL, n_test_static_policies=3, n_test_knn_policies=1)
    assert set(e.groups) == {"test_static", "test_knn"}
    assert e.total_calls == 4 * len(VAL)


def test_hand_check_of_cost_formula() -> None:
    e = est(GuideConfig(n_rounds=1, n_eval=1, n_init=1, phase2=False))
    g = e.groups["grade"]
    expected_high = g.calls * (g.input_tokens_per_call * 0.15 + g.output_tokens_per_call * 0.60) / 1e6
    rp = e.groups["rationale_plain"]
    expected_high += rp.calls * (rp.input_tokens_per_call * 0.15 + rp.output_tokens_per_call * 0.60) / 1e6
    assert e.usd_high == pytest.approx(expected_high)
