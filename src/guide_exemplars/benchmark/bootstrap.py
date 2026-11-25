"""Percentile bootstrap over test items, with paired deltas between methods.

A method can have several prediction vectors (e.g. Random's independent draws). Its metric on a resample is the
mean across vectors, so draw-to-draw variance is averaged in and not mistaken for item variance. Paired
deltas reuse the same item indices for both methods.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from guide_exemplars.core.metrics import compute_metrics

METRICS = ("acc", "qwk", "adj_err", "nonadj_err")
PredVec = Sequence[int | None]


def _metric_values(gold: np.ndarray, preds: Sequence[np.ndarray], idx: np.ndarray, n_labels: int) -> dict:
    out = {m: [] for m in METRICS}
    g = gold[idx].tolist()
    for p in preds:
        pv = [None if x < 0 else int(x) for x in p[idx]]
        m = compute_metrics(g, pv, n_labels)
        for k in METRICS:
            out[k].append(getattr(m, k))
    return {
        k: float(np.nanmean(v)) if not all(math.isnan(x) for x in v) else math.nan for k, v in out.items()
    }


def _encode(preds: Sequence[PredVec]) -> list[np.ndarray]:
    return [np.array([-1 if x is None else x for x in p], dtype=int) for p in preds]


def bootstrap(
    gold: Sequence[int],
    preds: Sequence[PredVec],
    n_labels: int,
    B: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """{metric: {point, lo, hi, n_nan}} for one method."""
    gold_arr = np.asarray(gold, dtype=int)
    enc = _encode(preds)
    n = len(gold_arr)
    point = _metric_values(gold_arr, enc, np.arange(n), n_labels)
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in METRICS}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals = _metric_values(gold_arr, enc, idx, n_labels)
        for m in METRICS:
            samples[m].append(vals[m])
    alpha = (1 - ci) / 2
    out = {}
    for m in METRICS:
        arr = np.array(samples[m])
        ok = arr[~np.isnan(arr)]
        lo, hi = np.quantile(ok, [alpha, 1 - alpha]) if len(ok) else (math.nan, math.nan)
        out[m] = {"point": point[m], "lo": float(lo), "hi": float(hi), "n_nan": int(np.isnan(arr).sum())}
    return out


def paired_delta(
    gold: Sequence[int],
    a: Sequence[PredVec],
    b: Sequence[PredVec],
    n_labels: int,
    B: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """{metric: {delta, lo, hi, p_gt0}} for a − b on identical resamples."""
    gold_arr = np.asarray(gold, dtype=int)
    ea, eb = _encode(a), _encode(b)
    n = len(gold_arr)
    full = np.arange(n)
    pa, pb = _metric_values(gold_arr, ea, full, n_labels), _metric_values(gold_arr, eb, full, n_labels)
    rng = np.random.default_rng(seed)
    deltas = {m: [] for m in METRICS}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        va, vb = _metric_values(gold_arr, ea, idx, n_labels), _metric_values(gold_arr, eb, idx, n_labels)
        for m in METRICS:
            deltas[m].append(va[m] - vb[m])
    alpha = (1 - ci) / 2
    out = {}
    for m in METRICS:
        arr = np.array(deltas[m])
        ok = arr[~np.isnan(arr)]
        lo, hi = np.quantile(ok, [alpha, 1 - alpha]) if len(ok) else (math.nan, math.nan)
        out[m] = {
            "delta": pa[m] - pb[m],
            "lo": float(lo),
            "hi": float(hi),
            "p_gt0": float((ok > 0).mean()) if len(ok) else math.nan,
        }
    return out
