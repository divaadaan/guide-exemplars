"""Bayesian optimization over exemplar subsets: GP surrogate, expected improvement, final selection.

Observations keep raw quantities (correct count, size, contrastive score). Each suggestion re-scalarizes the
whole history with the current weights and the current Acc* (ParEGO-style), so GP targets never mix stale
scalarizations.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from guide_exemplars.core.contrastive import Candidate
from guide_exemplars.core.featurize import SetFeaturizer
from guide_exemplars.core.objective import Weights, objective
from guide_exemplars.core.types import Observation

_TIE_EPS = 1e-12


@dataclass(frozen=True)
class BOConfig:
    xi: float = 0.01
    n_restarts: int = 3
    length_scale: float = 1.0
    noise_level: float = 0.1
    alpha: float = 1e-6


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    """EI for maximization. Zero wherever the predictive std is (numerically) zero."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    improvement = mu - best - xi
    ei = np.zeros_like(mu)
    pos = sigma > 1e-12
    z = improvement[pos] / sigma[pos]
    ei[pos] = improvement[pos] * norm.cdf(z) + sigma[pos] * norm.pdf(z)
    return np.maximum(ei, 0.0)


def accuracy_of(obs: Observation) -> Fraction:
    return Fraction(obs.n_correct, obs.n_total)


class BayesianSelector:
    """Holds one round's observations and proposes the next subset to evaluate."""

    def __init__(self, featurizer: SetFeaturizer, random_state: int, cfg: BOConfig | None = None):
        self.featurizer = featurizer
        self.random_state = random_state
        self.cfg = cfg or BOConfig()
        self.observations: list[Observation] = []

    def observe(self, obs: Observation) -> None:
        self.observations.append(obs)

    def acc_star(self) -> float:
        return float(max(accuracy_of(o) for o in self.observations)) if self.observations else 0.0

    def scalarize(self, w: Weights) -> np.ndarray:
        acc_star = self.acc_star()
        return np.array(
            [objective(float(accuracy_of(o)), acc_star, o.size, o.contrastive, w) for o in self.observations]
        )

    def _new_gp(self) -> GaussianProcessRegressor:
        kernel = Matern(nu=2.5, length_scale=self.cfg.length_scale) + WhiteKernel(self.cfg.noise_level)
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.cfg.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.cfg.n_restarts,
            random_state=self.random_state,
        )

    def suggest(
        self,
        candidates: Sequence[Candidate],
        w: Weights,
        rng: np.random.Generator,
    ) -> tuple[Candidate, float]:
        """Candidate with the highest EI under weights ``w``, with ties broken by ``rng``.

        With fewer than two observations there is nothing to fit: a random candidate is returned, EI NaN.
        """
        if not candidates:
            raise ValueError("no candidates to choose from")
        if len(self.observations) < 2:
            return candidates[int(rng.integers(len(candidates)))], math.nan

        x_obs = self.featurizer.transform([o.subset for o in self.observations])
        x_cand = self.featurizer.transform([c.subset for c in candidates])
        mean = x_obs.mean(axis=0)
        std = x_obs.std(axis=0)
        std[std < 1e-12] = 1.0
        y = self.scalarize(w)

        gp = self._new_gp()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit((x_obs - mean) / std, y)
            mu, sigma = gp.predict((x_cand - mean) / std, return_std=True)

        ei = expected_improvement(mu, sigma, float(y.max()), self.cfg.xi)
        best = float(ei.max())
        ties = np.flatnonzero(ei >= best - _TIE_EPS)
        pick = int(ties[int(rng.integers(len(ties)))])
        return candidates[pick], float(ei[pick])


def select_final(observations: Sequence[Observation]) -> Observation:
    """Lexicographic pick: highest accuracy, then highest contrastive score, then smallest size, then key.

    Accuracy is compared as an exact fraction, so float rounding never decides a tie.
    """
    if not observations:
        raise ValueError("no observations to select from")
    return min(
        observations,
        key=lambda o: (-accuracy_of(o), -o.contrastive, o.size, o.subset.key()),
    )
