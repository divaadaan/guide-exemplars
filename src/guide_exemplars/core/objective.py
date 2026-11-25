"""GUIDE's scalarized multi-objective: accuracy, sparsity and contrastive density (paper Eq., verbatim).

    g(E) = max(w1 · (Acc(E) − Acc*), w2 · (−|E|)) + w3 · C(E)

with w1 ~ U(0.25, 1), w2 = 0.8 (1 − w1), w3 = 0.2 (1 − w1), resampled at each BO iteration.

|E| is the raw set size, as in the paper. It is deliberately *not* divided by k_max: with raw sizes (4–16) the
sparsity term is far more negative than any accuracy gap (≥ −1) unless w1 is close to 1. So the max follows
accuracy for competitive sets, and for sets far behind Acc* it flattens to a preference for smaller sets.
Dividing by k_max (a tempting "normalization") shrinks the sparsity term until it often outweighs accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Weights:
    w1: float
    w2: float
    w3: float


@dataclass(frozen=True)
class ObjectiveConfig:
    w1_range: tuple[float, float] = (0.25, 1.0)
    w2_ratio: float = 0.8
    w3_ratio: float = 0.2
    w3_override: float | None = None  # e.g. 0.0 for the w3-zero ablation


def weights_for(w1: float, cfg: ObjectiveConfig | None = None) -> Weights:
    cfg = cfg or ObjectiveConfig()
    w3 = cfg.w3_ratio * (1.0 - w1) if cfg.w3_override is None else cfg.w3_override
    return Weights(w1=w1, w2=cfg.w2_ratio * (1.0 - w1), w3=w3)


def sample_weights(rng: np.random.Generator, cfg: ObjectiveConfig | None = None) -> Weights:
    cfg = cfg or ObjectiveConfig()
    lo, hi = cfg.w1_range
    return weights_for(float(rng.uniform(lo, hi)), cfg)


def objective(acc: float, acc_star: float, size: int, contrastive: float, w: Weights) -> float:
    tchebycheff = max(w.w1 * (acc - acc_star), w.w2 * (-size))
    return tchebycheff + w.w3 * contrastive
