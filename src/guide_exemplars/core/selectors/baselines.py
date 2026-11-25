"""Baselines from the paper: Naive (fixed), Random, KNN-SBERT (per-query) and Vote-k (static, diverse).

All of them draw from the Phase-0 pool, so they use the same plain rationales GUIDE starts from. Only selection
(and GUIDE's refinement) differs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from guide_exemplars.core.selectors.base import SelectionContext, StaticPolicy, one_per_item
from guide_exemplars.core.types import Exemplar, Item


def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class NaiveSelector:
    """A fixed, label-balanced set: round-robin over labels, items in a fixed hash order. It ignores the run
    seed, so "fixed" really means fixed."""

    name = "naive"

    async def fit(self, ctx: SelectionContext) -> StaticPolicy:
        candidates = sorted(one_per_item(ctx.pool), key=lambda e: _stable_hash(e.item_id))
        by_label: dict[str, list[Exemplar]] = {}
        for e in candidates:
            by_label.setdefault(e.label, []).append(e)
        labels = sorted(by_label)
        chosen: list[Exemplar] = []
        depth = 0
        while len(chosen) < ctx.k and any(depth < len(v) for v in by_label.values()):
            for lbl in labels:
                if depth < len(by_label[lbl]) and len(chosen) < ctx.k:
                    chosen.append(by_label[lbl][depth])
            depth += 1
        return StaticPolicy(self.name, tuple(chosen))


class RandomSelector:
    name = "random"

    async def fit(self, ctx: SelectionContext) -> StaticPolicy:
        candidates = one_per_item(ctx.pool)
        k = min(ctx.k, len(candidates))
        idx = ctx.rng.choice(len(candidates), size=k, replace=False)
        return StaticPolicy(self.name, tuple(candidates[i] for i in sorted(idx)))


async def _response_vectors(ctx: SelectionContext, exemplars: Sequence[Exemplar]) -> np.ndarray:
    if ctx.embedder is None:
        raise ValueError("this selector needs a response embedder")
    return np.asarray(await ctx.embedder.embed([e.text for e in exemplars]), dtype=np.float64)


class VoteKSelector:
    """Static diverse set by greedy max-min (farthest-point) selection over response embeddings, starting
    from the medoid. The GUIDE paper describes Vote-k this way; the original Vote-k uses a graph-based score."""

    name = "vote_k"

    async def fit(self, ctx: SelectionContext) -> StaticPolicy:
        candidates = one_per_item(ctx.pool)
        vecs = await _response_vectors(ctx, candidates)
        sims = vecs @ vecs.T
        chosen = [int(np.argmax(sims.sum(axis=1)))]
        min_dist = 1.0 - sims[chosen[0]]
        while len(chosen) < min(ctx.k, len(candidates)):
            min_dist[chosen] = -np.inf
            nxt = int(np.argmax(min_dist))
            chosen.append(nxt)
            min_dist = np.minimum(min_dist, 1.0 - sims[nxt])
        return StaticPolicy(self.name, tuple(candidates[i] for i in chosen))


@dataclass
class KNNPolicy:
    """Per-query top-k pool exemplars by cosine similarity of response embeddings (distinct items)."""

    name: str
    exemplars: list[Exemplar]
    vectors: np.ndarray
    k: int
    embedder: object
    kind: Literal["static", "per_query"] = "per_query"
    _query: dict[str, np.ndarray] = field(default_factory=dict)

    async def prepare(self, items: Sequence[Item]) -> None:
        todo = [it for it in items if it.item_id not in self._query]
        if todo:
            vecs = np.asarray(await self.embedder.embed([it.text for it in todo]), dtype=np.float64)
            self._query.update({it.item_id: v for it, v in zip(todo, vecs, strict=True)})

    def demos_for(self, item: Item) -> Sequence[Exemplar]:
        q = self._query.get(item.item_id)
        if q is None:
            raise KeyError(f"KNN policy not prepared for item {item.item_id!r}")
        sims = self.vectors @ q
        order = np.argsort(-sims, kind="stable")
        out = [self.exemplars[i] for i in order if self.exemplars[i].item_id != item.item_id]
        return out[: self.k]


class KNNSelector:
    name = "knn"

    async def fit(self, ctx: SelectionContext) -> KNNPolicy:
        candidates = one_per_item(ctx.pool)
        vecs = await _response_vectors(ctx, candidates)
        return KNNPolicy(self.name, candidates, vecs, ctx.k, ctx.embedder)


BASELINES = {
    "naive": NaiveSelector,
    "random": RandomSelector,
    "knn": KNNSelector,
    "vote_k": VoteKSelector,
}
