"""Candidate pool of exemplars, with exact-duplicate merging, capping and subset sampling.

The pool keeps insertion order, so every iteration and sample is deterministic given an RNG stream.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import numpy as np

from guide_exemplars.core.types import Exemplar, Subset


class Pool:
    """Exemplars indexed by uid, plus the item→variants mapping."""

    def __init__(self, exemplars: Iterable[Exemplar] = ()):
        self._by_uid: dict[str, Exemplar] = {}
        self._by_item: dict[str, list[str]] = {}
        self.merge(exemplars)

    # ---- access ---------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_uid)

    def __contains__(self, uid: object) -> bool:
        return uid in self._by_uid

    def __iter__(self) -> Iterator[Exemplar]:
        return iter(self._by_uid.values())

    def get(self, uid: str) -> Exemplar:
        return self._by_uid[uid]

    def uids(self) -> list[str]:
        return list(self._by_uid)

    def exemplars(self) -> list[Exemplar]:
        return list(self._by_uid.values())

    def item_ids(self) -> list[str]:
        return list(self._by_item)

    def variants(self, item_id: str) -> list[Exemplar]:
        return [self._by_uid[u] for u in self._by_item.get(item_id, [])]

    def experts(self) -> list[Exemplar]:
        return [e for e in self._by_uid.values() if e.is_expert]

    def resolve(self, subset: Subset) -> list[Exemplar]:
        """Exemplars of ``subset`` in stable (sorted uid) order."""
        return [self._by_uid[u] for u in subset]

    # ---- mutation -------------------------------------------------------------------------------

    def merge(self, new: Iterable[Exemplar]) -> int:
        """Add exemplars, skipping exact duplicates (same uid). Returns the number added.

        Raises ``ValueError`` if an exemplar's text or label disagrees with an existing variant of its item.
        """
        added = 0
        for ex in new:
            if ex.uid in self._by_uid:
                continue
            siblings = self._by_item.get(ex.item_id)
            if siblings:
                ref = self._by_uid[siblings[0]]
                if ref.text != ex.text or ref.label != ex.label:
                    raise ValueError(f"item {ex.item_id!r} has conflicting text or label across variants")
            self._by_uid[ex.uid] = ex
            self._by_item.setdefault(ex.item_id, []).append(ex.uid)
            added += 1
        return added

    def cap(self, n_max: int, protect: Iterable[str], rng: np.random.Generator) -> list[str]:
        """Shrink the pool to ``n_max`` exemplars. Returns the evicted uids.

        Experts and ``protect`` uids are always kept, even if that leaves the pool over ``n_max``. The other
        slots are filled uniformly at random from the remaining exemplars. Survivors keep insertion order.
        """
        if n_max <= 0:
            raise ValueError("n_max must be positive")
        if len(self) <= n_max:
            return []
        protected = {u for u in protect if u in self._by_uid}
        protected |= {e.uid for e in self.experts()}
        others = [u for u in self._by_uid if u not in protected]
        n_free = max(0, n_max - len(protected))
        if n_free >= len(others):
            return []
        keep_idx = rng.choice(len(others), size=n_free, replace=False) if n_free else []
        keep = protected | {others[i] for i in keep_idx}
        evicted = [u for u in self._by_uid if u not in keep]
        for uid in evicted:
            ex = self._by_uid.pop(uid)
            variants = self._by_item[ex.item_id]
            variants.remove(uid)
            if not variants:
                del self._by_item[ex.item_id]
        return evicted

    # ---- subsets --------------------------------------------------------------------------------

    def random_subset(
        self,
        rng: np.random.Generator,
        k_min: int,
        k_max: int,
        exclude_items: Iterable[str] = (),
    ) -> Subset:
        """Random valid subset: size ~ U[k_min, k_max], distinct items, one random variant per item."""
        if not 1 <= k_min <= k_max:
            raise ValueError(f"invalid size bounds k_min={k_min}, k_max={k_max}")
        excluded = set(exclude_items)
        items = [i for i in self._by_item if i not in excluded]
        if len(items) < k_min:
            raise ValueError(f"pool has {len(items)} eligible items, fewer than k_min={k_min}")
        size = int(rng.integers(k_min, min(k_max, len(items)) + 1))
        chosen = rng.choice(len(items), size=size, replace=False)
        uids = []
        for i in chosen:
            variants = self._by_item[items[i]]
            uids.append(variants[int(rng.integers(len(variants)))])
        return Subset.of(uids)

    def is_valid(self, subset: Subset, k_min: int, k_max: int) -> bool:
        """Size within bounds, every uid in the pool, and at most one variant per item."""
        if not k_min <= len(subset) <= k_max:
            return False
        if any(u not in self._by_uid for u in subset.uids):
            return False
        items = [self._by_uid[u].item_id for u in subset.uids]
        return len(set(items)) == len(items)

    def items_of(self, subset: Subset) -> set[str]:
        return {self._by_uid[u].item_id for u in subset.uids}

    # ---- persistence ----------------------------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        return [e.model_dump(mode="json", exclude={"uid"}) for e in self._by_uid.values()]

    @classmethod
    def from_records(cls, records: Sequence[dict[str, Any]]) -> Pool:
        return cls(Exemplar.model_validate(r) for r in records)
