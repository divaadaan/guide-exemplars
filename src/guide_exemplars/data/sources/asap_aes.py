"""ASAP-AES (Kaggle "asap-aes", ``training_set_rel3.tsv``): one essay set, one trait, two raters.

Label policies for combining the two raters' trait scores:

- ``min_drop_gap2`` (default): drop items whose raters differ by 2 or more; otherwise take the lower score,
  the level both raters agree is met.
- ``rater1``: rater 1 only.
- ``mean_round``: mean of the two, rounded half up.
- ``agree_only``: keep only items where both raters agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from guide_exemplars.core.metrics import quadratic_weighted_kappa

LabelPolicy = Literal["min_drop_gap2", "rater1", "mean_round", "agree_only"]


@dataclass(frozen=True)
class RawItem:
    item_id: str
    text: str
    label: int
    rater1: int
    rater2: int


@dataclass(frozen=True)
class SourceReport:
    n_raw: int
    n_kept: int
    n_dropped: int
    label_counts: dict[str, int]
    human_acc: float
    human_qwk: float

    def to_dict(self) -> dict:
        return {
            "n_raw": self.n_raw,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "label_counts": self.label_counts,
            "human_rater1_vs_rater2": {"acc": self.human_acc, "qwk": self.human_qwk},
        }


def combine(r1: int, r2: int, policy: LabelPolicy) -> int | None:
    if policy == "rater1":
        return r1
    if policy == "agree_only":
        return r1 if r1 == r2 else None
    if policy == "mean_round":
        return math.floor((r1 + r2) / 2 + 0.5)
    if policy == "min_drop_gap2":
        return None if abs(r1 - r2) >= 2 else min(r1, r2)
    raise ValueError(f"unknown label policy {policy!r}")


def human_agreement(r1: list[int], r2: list[int], n_labels: int) -> tuple[float, float]:
    if not r1:
        return math.nan, math.nan
    acc = sum(a == b for a, b in zip(r1, r2, strict=True)) / len(r1)
    return acc, quadratic_weighted_kappa(r1, r2, n_labels)


def load(
    path: str | Path,
    essay_set: int,
    trait: int,
    policy: LabelPolicy = "min_drop_gap2",
    score_range: tuple[int, int] = (0, 3),
) -> tuple[list[RawItem], SourceReport]:
    import pandas as pd

    df = pd.read_csv(path, sep="\t", encoding="latin-1")
    c1, c2 = f"rater1_trait{trait}", f"rater2_trait{trait}"
    for col in ("essay_id", "essay_set", "essay", c1, c2):
        if col not in df.columns:
            raise ValueError(f"{path} has no column {col!r}; is this ASAP-AES training_set_rel3.tsv?")
    df = df[df["essay_set"] == essay_set].dropna(subset=[c1, c2])
    lo, hi = score_range
    items: list[RawItem] = []
    for row in df.itertuples(index=False):
        r1, r2 = int(getattr(row, c1)), int(getattr(row, c2))
        if not (lo <= r1 <= hi and lo <= r2 <= hi):
            raise ValueError(f"essay {row.essay_id}: trait scores {r1},{r2} outside {score_range}")
        label = combine(r1, r2, policy)
        if label is None:
            continue
        text = " ".join(str(row.essay).split())
        items.append(RawItem(str(row.essay_id), text, label, r1, r2))
    n_labels = hi - lo + 1
    acc, qwk = human_agreement([i.rater1 - lo for i in items], [i.rater2 - lo for i in items], n_labels)
    counts = {str(v): sum(i.label == v for i in items) for v in range(lo, hi + 1)}
    report = SourceReport(len(df), len(items), len(df) - len(items), counts, acc, qwk)
    return items, report
