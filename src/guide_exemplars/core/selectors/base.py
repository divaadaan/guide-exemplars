"""Exemplar policies (what demonstrations each item is graded with) and the selectors that build them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from guide_exemplars.core.pool import Pool
from guide_exemplars.core.protocols import Embedder
from guide_exemplars.core.types import Exemplar, Item


class ExemplarPolicy(Protocol):
    name: str
    kind: Literal["static", "per_query"]

    async def prepare(self, items: Sequence[Item]) -> None:
        """Precompute anything per-item (e.g. query embeddings) before grading ``items``."""
        ...

    def demos_for(self, item: Item) -> Sequence[Exemplar]: ...


@dataclass
class StaticPolicy:
    name: str
    exemplars: tuple[Exemplar, ...]
    kind: Literal["static", "per_query"] = "static"

    async def prepare(self, items: Sequence[Item]) -> None:
        return None

    def demos_for(self, item: Item) -> Sequence[Exemplar]:
        return self.exemplars


@dataclass
class SelectionContext:
    pool: Pool  # Phase-0 pool: one rationale per item
    k: int
    rng: np.random.Generator
    embedder: Embedder | None = None  # response-text embedder for KNN / Vote-k
    metadata: dict = field(default_factory=dict)


class Selector(Protocol):
    name: str

    async def fit(self, ctx: SelectionContext) -> ExemplarPolicy: ...


def one_per_item(pool: Pool) -> list[Exemplar]:
    """First variant of each item, in pool order (the Phase-0 rationale)."""
    return [pool.variants(item_id)[0] for item_id in pool.item_ids()]
