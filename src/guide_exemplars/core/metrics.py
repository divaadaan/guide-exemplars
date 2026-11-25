"""Grading metrics on label ranks: accuracy, QWK, adjacent and non-adjacent error rates.

A failed prediction (``None``) counts as wrong for accuracy. It is left out of QWK, which needs a predicted
level, and is not classified as an adjacent or non-adjacent error. Every rate uses the full item count as its
denominator, so a demonstration set that causes parse failures cannot look more accurate than it is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from guide_exemplars.core.scale import LabelScale
    from guide_exemplars.core.types import Prediction


@dataclass(frozen=True)
class Metrics:
    acc: float
    qwk: float
    adj_err: float
    nonadj_err: float
    n: int
    n_correct: int
    n_failed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def quadratic_weighted_kappa(gold: Sequence[int], pred: Sequence[int], n_labels: int) -> float:
    """Cohen's kappa with quadratic weights over ranks ``0..n_labels-1``.

    Returns NaN when it is undefined: no items, or zero expected disagreement (e.g. every gold and predicted
    rank is the same single level).
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have equal length")
    if not gold or n_labels < 2:
        return math.nan
    g = np.asarray(gold, dtype=int)
    p = np.asarray(pred, dtype=int)
    observed = np.zeros((n_labels, n_labels), dtype=float)
    np.add.at(observed, (g, p), 1.0)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    idx = np.arange(n_labels)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (n_labels - 1) ** 2
    denom = float((weights * expected).sum())
    if denom == 0.0:
        return math.nan
    return 1.0 - float((weights * observed).sum()) / denom


def compute_metrics(gold: Sequence[int], pred: Sequence[int | None], n_labels: int) -> Metrics:
    """Metrics from aligned gold ranks and predicted ranks, where ``None`` marks a failure."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have equal length")
    n = len(gold)
    if n == 0:
        raise ValueError("cannot compute metrics on zero items")
    for r in (*gold, *(p for p in pred if p is not None)):
        if not 0 <= r < n_labels:
            raise ValueError(f"rank {r} outside 0..{n_labels - 1}")

    scored = [(g, p) for g, p in zip(gold, pred, strict=True) if p is not None]
    n_failed = n - len(scored)
    n_correct = sum(1 for g, p in scored if g == p)
    n_adj = sum(1 for g, p in scored if abs(g - p) == 1)
    n_nonadj = sum(1 for g, p in scored if abs(g - p) >= 2)
    qwk = quadratic_weighted_kappa([g for g, _ in scored], [p for _, p in scored], n_labels)
    return Metrics(
        acc=n_correct / n,
        qwk=qwk,
        adj_err=n_adj / n,
        nonadj_err=n_nonadj / n,
        n=n,
        n_correct=n_correct,
        n_failed=n_failed,
    )


def metrics_from_predictions(predictions: Sequence[Prediction], scale: LabelScale) -> Metrics:
    """Metrics from label-space predictions, mapped to ranks on ``scale``."""
    gold = [scale.rank(p.gold) for p in predictions]
    pred = [None if p.pred is None else scale.rank(p.pred) for p in predictions]
    return compute_metrics(gold, pred, len(scale))
