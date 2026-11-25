from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from guide_exemplars.core.metrics import (
    compute_metrics,
    metrics_from_predictions,
    quadratic_weighted_kappa,
)
from guide_exemplars.core.scale import LabelScale
from guide_exemplars.core.types import Prediction


def test_hand_computed_rates() -> None:
    gold = [0, 1, 2, 3, 3, 0]
    pred = [0, 2, 2, 1, None, 0]  # correct, adj, correct, non-adj, failed, correct
    m = compute_metrics(gold, pred, n_labels=4)
    assert m.n == 6 and m.n_correct == 3 and m.n_failed == 1
    assert m.acc == pytest.approx(3 / 6)
    assert m.adj_err == pytest.approx(1 / 6)
    assert m.nonadj_err == pytest.approx(1 / 6)


def test_failures_count_as_wrong_not_dropped() -> None:
    all_ok = compute_metrics([1, 1], [1, 1], 3)
    one_failed = compute_metrics([1, 1], [1, None], 3)
    assert all_ok.acc == 1.0
    assert one_failed.acc == 0.5


@pytest.mark.parametrize("seed", range(5))
def test_qwk_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 4, size=60).tolist()
    pred = np.clip(np.array(gold) + rng.integers(-1, 2, size=60), 0, 3).tolist()
    ours = quadratic_weighted_kappa(gold, pred, 4)
    theirs = cohen_kappa_score(gold, pred, weights="quadratic", labels=list(range(4)))
    assert ours == pytest.approx(theirs)


def test_qwk_excludes_failures() -> None:
    m = compute_metrics([0, 1, 2, 3], [0, 1, 2, None], 4)
    assert m.qwk == pytest.approx(quadratic_weighted_kappa([0, 1, 2], [0, 1, 2], 4))


def test_qwk_undefined_cases_are_nan() -> None:
    assert math.isnan(quadratic_weighted_kappa([], [], 4))
    assert math.isnan(quadratic_weighted_kappa([2, 2, 2], [2, 2, 2], 4))
    assert math.isnan(compute_metrics([1], [None], 3).qwk)


def test_named_labels_via_predictions() -> None:
    scale = LabelScale(("Beginning", "Developing", "Proficient"))
    preds = [
        Prediction("a", "Beginning", "Beginning"),
        Prediction("b", "Proficient", "Beginning"),
        Prediction("c", "Developing", None, error="parse"),
    ]
    m = metrics_from_predictions(preds, scale)
    assert (m.n_correct, m.n_failed) == (1, 1)
    assert m.nonadj_err == pytest.approx(1 / 3)
    assert m.adj_err == 0.0


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="zero items"):
        compute_metrics([], [], 3)
    with pytest.raises(ValueError, match="equal length"):
        compute_metrics([0], [0, 1], 3)
    with pytest.raises(ValueError, match="outside"):
        compute_metrics([0, 5], [0, 1], 3)


def test_to_dict() -> None:
    d = compute_metrics([0, 1], [0, 1], 2).to_dict()
    assert set(d) == {"acc", "qwk", "adj_err", "nonadj_err", "n", "n_correct", "n_failed"}
