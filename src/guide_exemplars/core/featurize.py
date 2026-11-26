"""Feature maps from exemplar subsets to vectors for the Gaussian-process surrogate.

``SetEmbeddingFeaturizer`` is the default. It describes a subset by the mean of its members'
PCA-projected embeddings, its label-rank histogram, and its relative size. Similar sets then sit close
together even when they share no members. That matters with only ~32 observations per round over a pool of
hundreds.

``MembershipFeaturizer`` is the paper's literal choice: one binary coordinate per pool exemplar (by uid).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA

from guide_exemplars.core.embedding_index import EmbeddingIndex
from guide_exemplars.core.pool import Pool
from guide_exemplars.core.scale import LabelScale
from guide_exemplars.core.types import Subset


class SetFeaturizer(Protocol):
    def fit(self, pool: Pool, index: EmbeddingIndex, scale: LabelScale, k_max: int) -> None: ...

    def transform(self, subsets: Sequence[Subset]) -> np.ndarray: ...


class MembershipFeaturizer:
    """Binary membership vector over the pool's uids (the paper's literal choice)."""

    def __init__(self) -> None:
        self._pos: dict[str, int] = {}

    def fit(self, pool: Pool, index: EmbeddingIndex, scale: LabelScale, k_max: int) -> None:
        self._pos = {u: i for i, u in enumerate(pool.uids())}

    def transform(self, subsets: Sequence[Subset]) -> np.ndarray:
        x = np.zeros((len(subsets), len(self._pos)), dtype=np.float64)
        for row, subset in enumerate(subsets):
            for uid in subset:
                x[row, self._pos[uid]] = 1.0
        return x


class SetEmbeddingFeaturizer:
    """[mean PCA embedding ‖ normalized label-rank histogram ‖ |E|/k_max]."""

    def __init__(self, n_components: int = 16):
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self._proj: dict[str, np.ndarray] = {}
        self._rank: dict[str, int] = {}
        self._n_labels = 0
        self._k_max = 1
        self.dim = 0

    def fit(self, pool: Pool, index: EmbeddingIndex, scale: LabelScale, k_max: int) -> None:
        uids = pool.uids()
        vectors = index.vectors(uids)
        n_comp = max(1, min(self.n_components, len(uids) - 1, vectors.shape[1]))
        projected = PCA(n_components=n_comp, svd_solver="full").fit_transform(vectors)
        self._proj = {u: projected[i] for i, u in enumerate(uids)}
        self._rank = {e.uid: scale.rank(e.label) for e in pool}
        self._n_labels = len(scale)
        self._k_max = k_max
        self.dim = n_comp + self._n_labels + 1

    def transform(self, subsets: Sequence[Subset]) -> np.ndarray:
        if not self._proj:
            raise RuntimeError("featurizer used before fit()")
        x = np.zeros((len(subsets), self.dim), dtype=np.float64)
        n_comp = self.dim - self._n_labels - 1
        for row, subset in enumerate(subsets):
            members = list(subset)
            if not members:
                continue
            x[row, :n_comp] = np.mean([self._proj[u] for u in members], axis=0)
            hist = np.bincount([self._rank[u] for u in members], minlength=self._n_labels)
            x[row, n_comp : n_comp + self._n_labels] = hist / len(members)
            x[row, -1] = len(members) / self._k_max
        return x
