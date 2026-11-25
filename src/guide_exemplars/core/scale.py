"""Ordered label scales for ordinal grading.

Labels are stored canonically as strings, so integer scales (``0..3``) and named scales
(``Beginning < Developing < Proficient``) share one code path. Adjacency is defined on rank, not numeric
value: on a non-uniform scale such as ``[0, 10, 20, 30]``, ``10`` and ``20`` are adjacent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

_STRIP_CHARS = " \t\r\n\"'`.,;:!?()[]{}"
_NUMERIC = re.compile(r"^[+-]?\d+(\.0+)?$")


def canonical_label(value: int | str) -> str:
    """Canonical string form of a label value. ``2``, ``"2"`` and ``"2.0"`` all become ``"2"``."""
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid label")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"non-integer numeric label: {value!r}")
        return str(int(value))
    text = str(value).strip()
    if _NUMERIC.match(text):
        return str(int(float(text)))
    return text


@dataclass(frozen=True)
class LabelScale:
    """An ordered set of labels, low to high."""

    labels: tuple[str, ...]
    descriptions: tuple[str, ...] = ()
    _rank: Mapping[str, int] = field(init=False, repr=False, compare=False, hash=False)
    _folded: Mapping[str, str] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        labels = tuple(canonical_label(lbl) for lbl in self.labels)
        if len(labels) < 2:
            raise ValueError("a label scale needs at least two labels")
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate labels in scale: {labels}")
        folded = {lbl.casefold(): lbl for lbl in labels}
        if len(folded) != len(labels):
            raise ValueError(f"labels differ only by case: {labels}")
        if self.descriptions and len(self.descriptions) != len(labels):
            raise ValueError("descriptions must align one-to-one with labels")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "descriptions", tuple(self.descriptions))
        object.__setattr__(self, "_rank", {lbl: i for i, lbl in enumerate(labels)})
        object.__setattr__(self, "_folded", folded)

    @classmethod
    def from_spec(cls, values: Iterable[int | str | Mapping[str, object]]) -> LabelScale:
        """Build from a list of bare values or ``{"value": ..., "description": ...}`` mappings."""
        labels: list[str] = []
        descriptions: list[str] = []
        for v in values:
            if isinstance(v, Mapping):
                labels.append(canonical_label(v["value"]))  # type: ignore[arg-type]
                descriptions.append(str(v.get("description", "") or ""))
            else:
                labels.append(canonical_label(v))
                descriptions.append("")
        return cls(tuple(labels), tuple(descriptions) if any(descriptions) else ())

    def __len__(self) -> int:
        return len(self.labels)

    def __contains__(self, label: object) -> bool:
        try:
            return canonical_label(label) in self._rank  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def rank(self, label: int | str) -> int:
        try:
            return self._rank[canonical_label(label)]
        except KeyError:
            raise ValueError(f"unknown label {label!r}; scale is {self.labels}") from None

    def label_at(self, rank: int) -> str:
        if not 0 <= rank < len(self.labels):
            raise IndexError(f"rank {rank} outside scale of size {len(self.labels)}")
        return self.labels[rank]

    def description(self, label: int | str) -> str:
        return self.descriptions[self.rank(label)] if self.descriptions else ""

    def neighbours(self, label: int | str) -> list[str]:
        """Labels one rank below and above (one-sided at the ends of the scale)."""
        r = self.rank(label)
        return [self.labels[i] for i in (r - 1, r + 1) if 0 <= i < len(self.labels)]

    def distance(self, a: int | str, b: int | str) -> int:
        return abs(self.rank(a) - self.rank(b))

    def is_adjacent(self, a: int | str, b: int | str) -> bool:
        return self.distance(a, b) == 1

    def parse(self, raw: object) -> str | None:
        """Map raw model output to a label. Only exact matches count, after normalization.

        Normalization strips whitespace, surrounding quotes and punctuation, folds case, and turns numeric
        forms like ``"2.0"`` into ``"2"``. Returns ``None`` when nothing matches. Free text containing a label
        (e.g. ``"score: 2"``) is not accepted here; extracting the label is the caller's job.
        """
        if raw is None or isinstance(raw, bool):
            return None
        if isinstance(raw, int | float):
            try:
                candidate = canonical_label(raw)
            except ValueError:
                return None
            return candidate if candidate in self._rank else None
        text = str(raw).strip(_STRIP_CHARS)
        if not text:
            return None
        if _NUMERIC.match(text):
            text = str(int(float(text)))
        return self._folded.get(text.casefold())
