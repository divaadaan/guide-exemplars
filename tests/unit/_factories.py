"""Shared builders for unit tests (imported directly; pytest puts this directory on sys.path)."""

from __future__ import annotations

import math

import numpy as np

from guide_exemplars.core.embedding_index import EmbeddingIndex
from guide_exemplars.core.metrics import Metrics
from guide_exemplars.core.pool import Pool
from guide_exemplars.core.scale import LabelScale
from guide_exemplars.core.types import Exemplar, Observation, Subset

INT4 = LabelScale.from_spec([0, 1, 2, 3])


def ex(item_id: str, label: int | str, rationale: str = "r", *, expert: bool = False) -> Exemplar:
    return Exemplar(
        item_id=item_id, text=f"text {item_id}", label=label, rationale=rationale, is_expert=expert
    )


def angle_vec(degrees: float) -> list[float]:
    rad = math.radians(degrees)
    return [math.cos(rad), math.sin(rad)]


def build(spec: list[tuple[Exemplar, list[float]]]) -> tuple[Pool, EmbeddingIndex]:
    pool = Pool(e for e, _ in spec)
    index = EmbeddingIndex([e.uid for e, _ in spec], np.array([v for _, v in spec]))
    return pool, index


def dense_pool(
    n_items: int = 60, variants: int = 2, dim: int = 8, noise: float = 0.15, seed: int = 0
) -> tuple[Pool, EmbeddingIndex]:
    """Random pool whose embeddings are all fairly similar, so the boundary graph is dense."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    spec = []
    for i in range(n_items):
        label = int(rng.integers(0, 4))
        item_vec = base + rng.normal(scale=noise * np.linalg.norm(base), size=dim)
        for v in range(variants):
            jitter = rng.normal(scale=0.02 * np.linalg.norm(base), size=dim)
            spec.append((ex(f"i{i}", label, f"rationale {v}"), list(item_vec + jitter)))
    return build(spec)


def obs(
    subset: Subset, n_correct: int, n_total: int = 10, contrastive: float = 0.0, index: int = 0
) -> Observation:
    metrics = Metrics(
        acc=n_correct / n_total,
        qwk=math.nan,
        adj_err=0.0,
        nonadj_err=0.0,
        n=n_total,
        n_correct=n_correct,
        n_failed=0,
    )
    return Observation(
        round=1, index=index, subset=subset, op="init", metrics=metrics, contrastive=contrastive
    )
