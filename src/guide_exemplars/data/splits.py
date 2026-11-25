"""Seeded stratified splits and expert carving.

The paper uses train:val:test = 3:1:1. Public data has no expert rationales, so "experts" are bootstrapped:
``experts_per_label`` training items per label, drawn with a seeded RNG, whose Phase-0 rationales are written
without context and then protected like expert exemplars.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np

from guide_exemplars.core.scale import LabelScale
from guide_exemplars.core.types import Item

Splits = dict[str, str]  # item_id -> train | val | test | expert


def stratified_split(
    items: Sequence[Item],
    scale: LabelScale,
    ratios: Sequence[float] = (3, 1, 1),
    seed: int = 0,
    experts_per_label: int = 1,
) -> Splits:
    if len(ratios) != 3 or min(ratios) < 0 or sum(ratios) <= 0:
        raise ValueError("ratios must be three non-negative numbers (train, val, test)")
    total = float(sum(ratios))
    fractions = [r / total for r in ratios]
    rng = np.random.default_rng(seed)
    splits: Splits = {}
    for label in scale.labels:
        group = sorted((it for it in items if it.label == label), key=lambda it: it.item_id)
        order = rng.permutation(len(group))
        n = len(group)
        n_val = round(n * fractions[1])
        n_test = round(n * fractions[2])
        n_val, n_test = min(n_val, n), min(n_test, max(0, n - n_val))
        for rank, idx in enumerate(order):
            if rank < n_test:
                split = "test"
            elif rank < n_test + n_val:
                split = "val"
            else:
                split = "train"
            splits[group[idx].item_id] = split
    return carve_experts(items, splits, scale, experts_per_label, seed)


def carve_experts(
    items: Sequence[Item], splits: Splits, scale: LabelScale, per_label: int, seed: int
) -> Splits:
    if per_label <= 0:
        return dict(splits)
    out = dict(splits)
    rng = np.random.default_rng([seed, 1])
    for label in scale.labels:
        pool = sorted(it.item_id for it in items if it.label == label and out.get(it.item_id) == "train")
        if len(pool) <= per_label:
            continue  # keep at least one training item per label
        for i in sorted(rng.choice(len(pool), per_label, replace=False)):
            out[pool[i]] = "expert"
    return out


def split_hash(items: Sequence[Item], splits: Splits) -> str:
    payload = json.dumps(
        sorted((it.item_id, it.label, splits.get(it.item_id, "")) for it in items), separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def limit_split(items: Sequence[Item], n: int | None, seed: int, scale: LabelScale) -> list[Item]:
    """Deterministic, label-stratified subsample of at most ``n`` items (``None`` keeps everything)."""
    items = list(items)
    if n is None or n >= len(items):
        return items
    rng = np.random.default_rng([seed, 2])
    by_label = {lbl: [it for it in items if it.label == lbl] for lbl in scale.labels}
    quotas = {lbl: round(n * len(group) / len(items)) for lbl, group in by_label.items()}
    while sum(quotas.values()) > n:
        quotas[max(quotas, key=quotas.get)] -= 1
    while sum(quotas.values()) < n:
        quotas[max(by_label, key=lambda lbl: len(by_label[lbl]) - quotas[lbl])] += 1
    keep: set[str] = set()
    for lbl, group in by_label.items():
        q = min(quotas[lbl], len(group))
        keep |= {group[i].item_id for i in rng.choice(len(group), q, replace=False)}
    return [it for it in items if it.item_id in keep]
