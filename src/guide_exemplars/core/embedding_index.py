"""Exemplar embeddings with a precomputed cosine-similarity matrix.

Pools are capped at a few hundred exemplars, so a dense similarity matrix is cheap and keeps every
boundary-pair query an O(1) lookup.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class EmbeddingIndex:
    """Row-normalized embeddings keyed by exemplar uid."""

    def __init__(self, uids: Sequence[str], vectors: np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != len(uids):
            raise ValueError(f"expected ({len(uids)}, d) vectors, got {vectors.shape}")
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate uids in embedding index")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        self._uids = list(uids)
        self._pos = {u: i for i, u in enumerate(self._uids)}
        self._sim = self._vectors @ self._vectors.T

    def __len__(self) -> int:
        return len(self._uids)

    def __contains__(self, uid: object) -> bool:
        return uid in self._pos

    @property
    def uids(self) -> list[str]:
        return list(self._uids)

    @property
    def dim(self) -> int:
        return self._vectors.shape[1]

    def position(self, uid: str) -> int:
        try:
            return self._pos[uid]
        except KeyError:
            raise KeyError(f"uid {uid!r} not in embedding index") from None

    def positions(self, uids: Sequence[str]) -> np.ndarray:
        return np.fromiter((self.position(u) for u in uids), dtype=np.intp, count=len(uids))

    def sim(self, a: str, b: str) -> float:
        return float(self._sim[self.position(a), self.position(b)])

    def sim_matrix(self, uids: Sequence[str] | None = None) -> np.ndarray:
        if uids is None:
            return self._sim.copy()
        idx = self.positions(uids)
        return self._sim[np.ix_(idx, idx)]

    def vectors(self, uids: Sequence[str] | None = None) -> np.ndarray:
        if uids is None:
            return self._vectors.copy()
        return self._vectors[self.positions(uids)]
