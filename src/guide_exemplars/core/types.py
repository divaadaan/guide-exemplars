"""Value types shared across the core: items, exemplars, subsets, predictions, observations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from guide_exemplars.core.metrics import Metrics
from guide_exemplars.core.scale import canonical_label

Origin = Literal["given", "phase0", "phase2"]


def normalize_rationale(text: str) -> str:
    """Normalization used for exact-duplicate detection: collapse whitespace, fold case."""
    return " ".join(text.split()).casefold()


class Item(BaseModel):
    """A labeled response: one row of a dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    text: str
    label: str

    @field_validator("item_id", mode="before")
    @classmethod
    def _id_to_str(cls, v: object) -> str:
        return str(v)

    @field_validator("label", mode="before")
    @classmethod
    def _label_to_canonical(cls, v: object) -> str:
        return canonical_label(v)  # type: ignore[arg-type]


class Exemplar(BaseModel):
    """A candidate demonstration: an item plus a rationale for its label.

    One item can have several exemplars (rationale variants from different rounds), so identity is the
    ``uid``, never the ``item_id``.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    item_id: str
    text: str
    label: str
    rationale: str
    is_expert: bool = False
    origin: Origin = "phase0"
    round: int = 0

    @field_validator("item_id", mode="before")
    @classmethod
    def _id_to_str(cls, v: object) -> str:
        return str(v)

    @field_validator("label", mode="before")
    @classmethod
    def _label_to_canonical(cls, v: object) -> str:
        return canonical_label(v)  # type: ignore[arg-type]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def uid(self) -> str:
        digest = hashlib.sha256(normalize_rationale(self.rationale).encode("utf-8")).hexdigest()
        return f"{self.item_id}#{digest[:12]}"

    @classmethod
    def from_item(cls, item: Item, rationale: str, **kwargs: object) -> Exemplar:
        return cls(item_id=item.item_id, text=item.text, label=item.label, rationale=rationale, **kwargs)


@dataclass(frozen=True)
class Subset:
    """An unordered set of exemplar uids: one candidate demonstration set."""

    uids: frozenset[str]

    @classmethod
    def of(cls, uids: Iterable[str]) -> Subset:
        return cls(frozenset(uids))

    def key(self) -> str:
        """Order-independent content key, used for dedup, caching and seeding demo order."""
        joined = "\n".join(sorted(self.uids))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.uids)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.uids))

    def __contains__(self, uid: object) -> bool:
        return uid in self.uids


@dataclass(frozen=True)
class Prediction:
    """One grading outcome, aligned to its item. A failed call has ``pred=None`` and counts as wrong."""

    item_id: str
    gold: str
    pred: str | None
    raw: str = ""
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.pred is None

    @property
    def correct(self) -> bool:
        return self.pred is not None and self.pred == self.gold


@dataclass(frozen=True)
class Observation:
    """One evaluated subset in the optimization history. Raw quantities are kept so the objective can be
    re-scalarized with new weights at any time."""

    round: int
    index: int
    subset: Subset
    op: str
    metrics: Metrics
    contrastive: float
    ei: float | None = None

    @property
    def size(self) -> int:
        return len(self.subset)

    @property
    def n_correct(self) -> int:
        return self.metrics.n_correct

    @property
    def n_total(self) -> int:
        return self.metrics.n
