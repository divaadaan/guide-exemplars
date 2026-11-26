"""Feature grader: an LLM counts evidence in each response, and a small ordinal model turns the counts into a score.

The task spec's ``features`` block says what to count (for an essay-conventions trait: errors a reader corrects
without effort, and errors that block understanding). Extraction is zero-shot at temperature 0, one short call
per response. Counts become rates per ``per_words`` words, optionally with log word count, so a long essay is
not penalized for having more errors in total.

``OrdinalRidge`` fits a ridge regression from those features to the score rank, then picks cut points on the
regression score that maximize QWK on the fitting set. The hybrid grader adds GUIDE's predicted rank as one more
column, so the model learns how much to trust the LLM grader relative to the counted evidence.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from guide_exemplars.core.metrics import quadratic_weighted_kappa
from guide_exemplars.core.prompts import PromptBuilder, parse_counts
from guide_exemplars.core.protocols import (
    LLM,
    AbortRun,
    Cancelled,
    CancelToken,
    LLMRequest,
    NeverCancel,
    ProviderUnstable,
)
from guide_exemplars.core.types import Item


def word_count(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class Extraction:
    """Counts for one response (``None`` when the call or its parse failed) and the raw reply."""

    item_id: str
    counts: dict[str, int] | None
    raw: str
    error: str | None = None


class FeatureExtractor:
    def __init__(
        self,
        prompts: PromptBuilder,
        llm: LLM,
        max_tokens: int = 100,
        cancel: CancelToken | None = None,
        max_error_rate: float | None = 0.02,
        min_errors_to_abort: int = 3,
    ):
        spec = prompts.task.features
        if spec is None:
            raise ValueError(f"task {prompts.task.name!r} defines no features")
        self.prompts = prompts
        self.spec = spec
        self.llm = llm
        self.max_tokens = max_tokens
        self.cancel = cancel or NeverCancel()
        self.max_error_rate = max_error_rate
        self.min_errors_to_abort = min_errors_to_abort
        self.count_names = [c.name for c in spec.counts]

    @property
    def feature_names(self) -> list[str]:
        suffix = f"_per_{self.spec.per_words}w" if self.spec.per_words else ""
        names = [f"{n}{suffix}" for n in self.count_names]
        return names + (["log_words"] if self.spec.include_length else [])

    def request(self, text: str) -> LLMRequest:
        return LLMRequest.of(
            self.prompts.features_system,
            self.prompts.features_user(text),
            max_tokens=self.max_tokens,
            purpose="features",
            temperature=0.0,
        )

    async def _extract_one(self, item: Item) -> tuple[Extraction, bool]:
        if self.cancel.is_set():
            raise Cancelled("cancelled before feature call")
        try:
            res = await self.llm.complete(self.request(item.text))
        except AbortRun:
            raise
        except Exception as exc:
            return Extraction(item.item_id, None, "", f"{type(exc).__name__}: {exc}"), True
        counts = parse_counts(res.text, self.count_names)
        return Extraction(item.item_id, counts, res.text, None if counts else "unparsable counts"), False

    async def extract(self, items: Sequence[Item]) -> list[Extraction]:
        """Extractions in input order. A burst of provider errors aborts, as in grading."""
        outcomes = await asyncio.gather(*(self._extract_one(it) for it in items), return_exceptions=True)
        for out in outcomes:
            if isinstance(out, BaseException):
                raise out
        results = [out[0] for out in outcomes]  # type: ignore[index]
        n_err = sum(out[1] for out in outcomes)  # type: ignore[index]
        too_many = self.max_error_rate is not None and n_err > self.max_error_rate * len(items)
        if too_many and n_err >= self.min_errors_to_abort:
            raise ProviderUnstable(f"{n_err}/{len(items)} feature calls failed at the provider")
        return results

    def vectorize(self, items: Sequence[Item], extractions: Sequence[Extraction]) -> np.ndarray:
        """One row per item. Failed extractions are NaN (imputed by the model)."""
        rows = []
        for it, ex in zip(items, extractions, strict=True):
            words = word_count(it.text)
            if ex.counts is None:
                row = [math.nan] * len(self.count_names)
            elif self.spec.per_words:
                row = [ex.counts[n] / max(words, 1) * self.spec.per_words for n in self.count_names]
            else:
                row = [float(ex.counts[n]) for n in self.count_names]
            if self.spec.include_length:
                row.append(math.log1p(words))
            rows.append(row)
        return np.asarray(rows, dtype=float).reshape(len(items), len(self.feature_names))


def _qwk(y: np.ndarray, pred: np.ndarray, n_labels: int) -> float:
    k = quadratic_weighted_kappa(y.tolist(), pred.tolist(), n_labels)
    return -math.inf if math.isnan(k) else k


def apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Rank = number of cut points at or below the score."""
    return np.searchsorted(np.asarray(thresholds), np.asarray(scores, dtype=float), side="right")


def fit_thresholds(
    scores: Sequence[float], y: Sequence[int], n_labels: int, n_grid: int = 64, sweeps: int = 6
) -> np.ndarray:
    """Cut points on a continuous score that maximize QWK, by coordinate ascent.

    Starts from the cut points that reproduce the gold label proportions, then moves one cut point at a time
    to the best of ``n_grid`` score quantiles between its neighbours, until a sweep changes nothing.
    """
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int)
    props = np.cumsum(np.bincount(yy, minlength=n_labels))[:-1] / len(yy)
    t = np.maximum.accumulate(np.quantile(s, props))
    grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_grid + 1)))
    best = _qwk(yy, apply_thresholds(s, t), n_labels)
    for _ in range(sweeps):
        improved = False
        for j in range(n_labels - 1):
            lo = t[j - 1] if j > 0 else -math.inf
            hi = t[j + 1] if j < n_labels - 2 else math.inf
            for c in grid[(grid >= lo) & (grid <= hi)]:
                trial = t.copy()
                trial[j] = c
                k = _qwk(yy, apply_thresholds(s, trial), n_labels)
                if k > best + 1e-12:
                    best, t, improved = k, trial, True
        if not improved:
            break
    return t


class OrdinalRidge:
    """Ridge regression to the score rank, with QWK-fitted cut points. NaNs are imputed with fitting means."""

    def __init__(self, n_labels: int, alpha: float = 1.0):
        if n_labels < 2:
            raise ValueError("need at least two labels")
        self.n_labels = n_labels
        self.alpha = alpha

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = np.where(np.isnan(X), self.mean_, X)
        return (X - self.mean_) / self.scale_

    def fit(self, X: np.ndarray, y: Sequence[int]) -> OrdinalRidge:
        X = np.asarray(X, dtype=float)
        yy = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) != len(yy) or len(yy) < 2:
            raise ValueError("X must be 2-D with one row per label")
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(X, axis=0)
        self.mean_ = np.where(np.isnan(mean), 0.0, mean)
        filled = np.where(np.isnan(X), self.mean_, X)
        scale = filled.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        Z = (filled - self.mean_) / self.scale_
        self.intercept_ = float(yy.mean())
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (yy - self.intercept_))
        fit_scores = Z @ self.coef_ + self.intercept_
        self.thresholds_ = fit_thresholds(fit_scores, yy.astype(int), self.n_labels)
        self.fit_qwk_ = _qwk(yy.astype(int), apply_thresholds(fit_scores, self.thresholds_), self.n_labels)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._standardize(X) @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return apply_thresholds(self.score(X), self.thresholds_)

    def describe(self, feature_names: Sequence[str]) -> dict:
        """Standardized coefficients (comparable across features), cut points and fitting-set QWK."""
        return {
            "coef_standardized": {
                n: round(float(c), 4) for n, c in zip(feature_names, self.coef_, strict=True)
            },
            "thresholds": [round(float(t), 4) for t in self.thresholds_],
            "fit_qwk": round(float(self.fit_qwk_), 4),
        }


def extraction_raw(ex: Extraction) -> str:
    """Compact audit string for a prediction row."""
    return json.dumps(ex.counts) if ex.counts is not None else ex.raw
