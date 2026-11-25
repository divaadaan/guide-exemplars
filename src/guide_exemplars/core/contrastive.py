"""Boundary pairs, the contrastive score, and GUIDE's subset operators.

A boundary pair is two exemplars from different items whose embeddings have cosine similarity ≥ τ and
whose labels differ. The operators (Contrastive-Add, Contrastive-Swap) always use *adjacent* labels, which
is the paper's candidate set C_i. Whether the score also requires adjacency is set by
``ContrastiveConfig.adjacent_only``: ``True`` is the improved default, ``False`` is the paper's literal
definition.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from guide_exemplars.core.embedding_index import EmbeddingIndex
from guide_exemplars.core.pool import Pool
from guide_exemplars.core.scale import LabelScale
from guide_exemplars.core.types import Subset

# Absorbs float error so that a pair at exactly τ counts as a boundary pair.
_SIM_EPS = 1e-9

ScoreKind = Literal["coverage", "pair_ratio"]
OpName = Literal["add", "swap", "flip", "random", "expert", "init", "incumbent"]


@dataclass(frozen=True)
class ContrastiveConfig:
    tau: float = 0.7
    adjacent_only: bool = True
    score: ScoreKind = "coverage"


@dataclass(frozen=True)
class BoundaryPair:
    a: str
    b: str
    similarity: float
    distance: int


@dataclass(frozen=True)
class OperatorMix:
    """Per-step candidate quotas. Random fills whatever remains of the budget (minus the expert slot)."""

    add: int = 64
    swap: int = 64
    flip: int = 64


@dataclass(frozen=True)
class Candidate:
    subset: Subset
    op: OpName


class ContrastiveOps:
    """Boundary graph over one round's pool, plus the operators that perturb a subset along it."""

    def __init__(
        self,
        pool: Pool,
        index: EmbeddingIndex,
        scale: LabelScale,
        cfg: ContrastiveConfig,
        k_min: int,
        k_max: int,
    ):
        if not 1 <= k_min <= k_max:
            raise ValueError(f"invalid size bounds k_min={k_min}, k_max={k_max}")
        self.pool = pool
        self.index = index
        self.scale = scale
        self.cfg = cfg
        self.k_min = k_min
        self.k_max = k_max

        self._uids = pool.uids()
        self._pos = {u: i for i, u in enumerate(self._uids)}
        exemplars = pool.exemplars()
        self._items = [e.item_id for e in exemplars]
        self._experts = {e.uid for e in exemplars if e.is_expert}

        ranks = np.array([scale.rank(e.label) for e in exemplars])
        items = np.array(self._items, dtype=object)
        sim = index.sim_matrix(self._uids)
        rank_dist = np.abs(ranks[:, None] - ranks[None, :])
        close = (sim >= cfg.tau - _SIM_EPS) & (items[:, None] != items[None, :])

        self._sim = sim
        self._rank_dist = rank_dist
        # Operator neighbours: the paper's C_i (adjacent labels only).
        op_mask = close & (rank_dist == 1)
        self._op_neighbours = [np.flatnonzero(op_mask[i]) for i in range(len(self._uids))]
        # Score / reporting pairs: adjacent-only or any differing label, per config.
        self._pair_mask = close & ((rank_dist == 1) if cfg.adjacent_only else (rank_dist >= 1))

    # ---- boundary structure ---------------------------------------------------------------------

    def boundary_pairs(self, uids: Iterable[str] | None = None) -> list[BoundaryPair]:
        """All boundary pairs among ``uids`` (default: the whole pool), using the score's pair rule."""
        idx = list(range(len(self._uids))) if uids is None else [self._pos[u] for u in uids]
        pairs = []
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1 :]:
                if self._pair_mask[i, j]:
                    pairs.append(
                        BoundaryPair(
                            a=self._uids[i],
                            b=self._uids[j],
                            similarity=float(self._sim[i, j]),
                            distance=int(self._rank_dist[i, j]),
                        )
                    )
        return pairs

    def candidates_for(self, uid: str, exclude_items: Iterable[str] = ()) -> list[str]:
        """C_i: pool exemplars from other items, one label away and similar enough to ``uid``."""
        excluded = set(exclude_items)
        return [self._uids[j] for j in self._op_neighbours[self._pos[uid]] if self._items[j] not in excluded]

    def score(self, subset: Subset) -> float:
        """Contrastive score of a subset, in [0, 1].

        ``coverage`` is the fraction of members that have at least one boundary partner inside the subset.
        ``pair_ratio`` is the number of boundary pairs over C(n, 2).
        """
        n = len(subset)
        if n < 2:
            return 0.0
        idx = np.array([self._pos[u] for u in subset])
        block = self._pair_mask[np.ix_(idx, idx)]
        if self.cfg.score == "coverage":
            return float(block.any(axis=1).sum()) / n
        return float(np.triu(block, k=1).sum()) / (n * (n - 1) / 2)

    # ---- operators ------------------------------------------------------------------------------

    def _items_in(self, subset: Subset) -> set[str]:
        return {self._items[self._pos[u]] for u in subset}

    def add(self, subset: Subset, rng: np.random.Generator) -> Subset | None:
        """Contrastive-Add: insert a boundary neighbour of a random member."""
        if len(subset) >= self.k_max:
            return None
        members = list(subset)
        taken = self._items_in(subset)
        for i in rng.permutation(len(members)):
            cands = self.candidates_for(members[i], exclude_items=taken)
            if cands:
                return Subset.of([*members, cands[int(rng.integers(len(cands)))]])
        return None

    def swap(self, subset: Subset, rng: np.random.Generator) -> Subset | None:
        """Contrastive-Swap: replace a non-expert member with one of its boundary neighbours."""
        members = list(subset)
        taken = self._items_in(subset)
        for i in rng.permutation(len(members)):
            uid = members[i]
            if uid in self._experts:
                continue
            cands = self.candidates_for(uid, exclude_items=taken)
            if cands:
                replacement = cands[int(rng.integers(len(cands)))]
                return Subset.of([*(m for m in members if m != uid), replacement])
        return None

    def flip(self, subset: Subset, rng: np.random.Generator) -> Subset | None:
        """One-Flip: replace a random non-expert member with a random exemplar of an unused item."""
        members = list(subset)
        removable = [m for m in members if m not in self._experts]
        if not removable:
            return None
        taken = self._items_in(subset)
        available = [u for u, item in zip(self._uids, self._items, strict=True) if item not in taken]
        if not available:
            return None
        out = removable[int(rng.integers(len(removable)))]
        replacement = available[int(rng.integers(len(available)))]
        return Subset.of([*(m for m in members if m != out), replacement])

    def random(self, rng: np.random.Generator) -> Subset:
        return self.pool.random_subset(rng, self.k_min, self.k_max)

    # ---- candidate generation -------------------------------------------------------------------

    def generate_candidates(
        self,
        incumbent: Subset,
        expert: Subset | None,
        evaluated: Iterable[str],
        rng: np.random.Generator,
        n: int = 256,
        mix: OperatorMix | None = None,
        max_attempts_factor: int = 4,
    ) -> list[Candidate]:
        """Build up to ``n`` unique, valid, not-yet-evaluated candidate subsets for EI acquisition.

        Order and quotas: the expert subset (one slot, if valid and new), then ``mix.add`` / ``mix.swap``
        / ``mix.flip`` perturbations of the incumbent, then random subsets for the rest. A quota an operator
        cannot fill passes to the next one (Add → Swap → Flip → Random). Fewer than ``n`` come back only when
        the pool is too small to supply that many distinct subsets.
        """
        mix = mix or OperatorMix()
        if mix.add + mix.swap + mix.flip > n:
            raise ValueError("operator quotas exceed the candidate budget")
        seen = set(evaluated)
        out: list[Candidate] = []

        def accept(subset: Subset | None, op: OpName) -> bool:
            if subset is None or not self.pool.is_valid(subset, self.k_min, self.k_max):
                return False
            key = subset.key()
            if key in seen:
                return False
            seen.add(key)
            out.append(Candidate(subset, op))
            return True

        if expert is not None and len(expert):
            accept(expert, "expert")

        random_quota = n - len(out) - (mix.add + mix.swap + mix.flip)
        plan: Sequence[tuple[OpName, int]] = (
            ("add", mix.add),
            ("swap", mix.swap),
            ("flip", mix.flip),
            ("random", random_quota),
        )
        carry = 0
        for op, quota in plan:
            target = quota + carry
            got = 0
            for _ in range(max_attempts_factor * target):
                if got == target:
                    break
                if op == "add":
                    sub = self.add(incumbent, rng)
                elif op == "swap":
                    sub = self.swap(incumbent, rng)
                elif op == "flip":
                    sub = self.flip(incumbent, rng)
                else:
                    sub = self.random(rng)
                got += accept(sub, op)
            carry = target - got
        return out
