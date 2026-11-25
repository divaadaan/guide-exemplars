from __future__ import annotations

import numpy as np
import pytest

from guide_exemplars.core.objective import ObjectiveConfig, objective, sample_weights, weights_for


def test_weight_relations() -> None:
    w = weights_for(0.4)
    assert w.w2 == pytest.approx(0.8 * 0.6)
    assert w.w3 == pytest.approx(0.2 * 0.6)
    assert weights_for(0.4, ObjectiveConfig(w3_override=0.0)).w3 == 0.0


def test_sampled_w1_in_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = sample_weights(rng)
        assert 0.25 <= w.w1 <= 1.0
        assert w.w2 + w.w3 == pytest.approx(1.0 - w.w1)


def test_accuracy_gap_dominates_for_competitive_sets() -> None:
    w = weights_for(0.5)  # w2 = 0.4, w3 = 0.1
    # gap term 0.5 * (0.6 - 0.8) = -0.1 beats sparsity 0.4 * -8 = -3.2
    assert objective(0.6, 0.8, 8, contrastive=0.5, w=w) == pytest.approx(-0.1 + 0.05)
    assert objective(0.8, 0.8, 16, contrastive=0.0, w=w) == pytest.approx(0.0)


def test_sparsity_takes_over_for_sets_far_behind_when_w1_high() -> None:
    w = weights_for(0.95)  # w2 = 0.04
    # gap term 0.95 * -0.7 = -0.665 is worse than sparsity 0.04 * -4 = -0.16
    assert objective(0.2, 0.9, 4, contrastive=0.0, w=w) == pytest.approx(-0.16)
    assert objective(0.2, 0.9, 4, 0.0, w) > objective(0.2, 0.9, 12, 0.0, w)


def test_contrastive_bonus_is_additive() -> None:
    w = weights_for(0.25)
    base = objective(0.7, 0.8, 6, contrastive=0.0, w=w)
    assert objective(0.7, 0.8, 6, contrastive=1.0, w=w) == pytest.approx(base + w.w3)
