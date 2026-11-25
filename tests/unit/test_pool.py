from __future__ import annotations

import numpy as np
import pytest

from guide_exemplars.core.pool import Pool
from guide_exemplars.core.rng import RngStreams
from guide_exemplars.core.types import Exemplar, Subset


def _ex(i: int, rationale: str = "r", *, label: str | None = None, expert: bool = False) -> Exemplar:
    return Exemplar(
        item_id=f"i{i}",
        text=f"text {i}",
        label=label if label is not None else str(i % 3),
        rationale=rationale,
        is_expert=expert,
    )


def _pool(n: int = 20, variants: int = 1) -> Pool:
    return Pool(_ex(i, f"r{v}") for i in range(n) for v in range(variants))


def test_merge_dedups_exact_duplicates() -> None:
    pool = Pool([_ex(0, "Same reason."), _ex(1)])
    assert pool.merge([_ex(0, "same   REASON."), _ex(0, "Other reason.")]) == 1
    assert len(pool) == 3
    assert len(pool.variants("i0")) == 2
    assert pool.item_ids() == ["i0", "i1"]


def test_merge_rejects_conflicting_variant() -> None:
    pool = Pool([_ex(0)])
    with pytest.raises(ValueError, match="conflicting"):
        pool.merge([_ex(0, "new", label="2")])


def test_cap_keeps_experts_and_protected_and_is_deterministic() -> None:
    def build() -> Pool:
        p = _pool(30, variants=2)
        p.merge([_ex(100, expert=True, label="0"), _ex(101, expert=True, label="1")])
        return p

    protect = {_ex(5, "r1").uid}
    p1, p2 = build(), build()
    ev1 = p1.cap(20, protect, RngStreams(1).get("cap"))
    ev2 = p2.cap(20, protect, RngStreams(1).get("cap"))
    assert ev1 == ev2
    assert len(p1) == 20
    assert all(e.uid in p1 for e in p1.experts())
    assert len(p1.experts()) == 2
    assert protect <= set(p1.uids())
    assert not set(ev1) & set(p1.uids())

    p3 = build()
    p3.cap(20, protect, RngStreams(2).get("cap"))
    assert p3.uids() != p1.uids()


def test_cap_never_evicts_protected_even_if_over_budget() -> None:
    pool = Pool([_ex(i, expert=True) for i in range(5)] + [_ex(10), _ex(11)])
    evicted = pool.cap(3, set(), np.random.default_rng(0))
    assert len(pool) == 5
    assert set(evicted) == {_ex(10).uid, _ex(11).uid}


def test_cap_noop_when_under_limit() -> None:
    pool = _pool(5)
    assert pool.cap(10, set(), np.random.default_rng(0)) == []
    assert len(pool) == 5


def test_cap_removes_items_that_lose_all_variants() -> None:
    pool = _pool(10)
    pool.cap(4, set(), np.random.default_rng(0))
    assert len(pool.item_ids()) == 4


def test_random_subset_distinct_items_within_bounds() -> None:
    pool = _pool(20, variants=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = pool.random_subset(rng, 4, 8)
        assert pool.is_valid(s, 4, 8)
        assert len(pool.items_of(s)) == len(s)


def test_random_subset_respects_exclusions_and_small_pools() -> None:
    pool = _pool(6)
    rng = np.random.default_rng(0)
    s = pool.random_subset(rng, 2, 16, exclude_items={"i0", "i1"})
    assert not pool.items_of(s) & {"i0", "i1"}
    assert len(s) <= 4
    with pytest.raises(ValueError, match="fewer than k_min"):
        pool.random_subset(rng, 5, 8, exclude_items={"i0", "i1"})
    with pytest.raises(ValueError, match="bounds"):
        pool.random_subset(rng, 0, 3)


def test_random_subset_deterministic_per_stream() -> None:
    pool = _pool(30, variants=2)
    a = [pool.random_subset(RngStreams(7).get("init", i), 4, 16) for i in range(5)]
    b = [pool.random_subset(RngStreams(7).get("init", i), 4, 16) for i in range(5)]
    assert a == b


def test_is_valid_rejects_two_variants_of_one_item() -> None:
    pool = _pool(10, variants=2)
    v = pool.variants("i0")
    assert not pool.is_valid(Subset.of([v[0].uid, v[1].uid, pool.variants("i1")[0].uid]), 2, 8)
    assert not pool.is_valid(Subset.of(["missing#0", v[0].uid]), 1, 8)
    assert not pool.is_valid(Subset.of([v[0].uid]), 2, 8)


def test_records_roundtrip() -> None:
    pool = _pool(5, variants=2)
    restored = Pool.from_records(pool.to_records())
    assert restored.uids() == pool.uids()
    assert restored.exemplars() == pool.exemplars()


def test_resolve_returns_sorted_exemplars() -> None:
    pool = _pool(5)
    s = Subset.of([pool.uids()[3], pool.uids()[1]])
    assert [e.uid for e in pool.resolve(s)] == sorted(s.uids)
