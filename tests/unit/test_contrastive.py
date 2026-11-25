"""Boundary pairs, contrastive score, operators and candidate generation.

The hand-built geometry puts items on the unit circle, so cosine similarity is cos(Δangle). With τ = 0.7,
items closer than ~45.6° are "close":

    A  label 0  at   0°     B  label 1  at  10°     C  label 2  at  20°
    D  label 1  at  90°     E  label 0  at 100°     F  label 3  at 180°

Adjacent boundary pairs: A–B, B–C, D–E. A–C is close but two levels apart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _factories import INT4, angle_vec, build, dense_pool, ex

from guide_exemplars.core.contrastive import ContrastiveConfig, ContrastiveOps, OperatorMix
from guide_exemplars.core.types import Subset

A, B, C = ex("A", 0), ex("B", 1), ex("C", 2)
D, E, F = ex("D", 1), ex("E", 0), ex("F", 3)
GEOMETRY = [(A, 0), (B, 10), (C, 20), (D, 90), (E, 100), (F, 180)]


def make_ops(
    extra: list | None = None,
    cfg: ContrastiveConfig | None = None,
    k_min: int = 1,
    k_max: int = 4,
) -> ContrastiveOps:
    spec = [(e, angle_vec(deg)) for e, deg in GEOMETRY] + (extra or [])
    pool, index = build(spec)
    return ContrastiveOps(pool, index, INT4, cfg or ContrastiveConfig(), k_min, k_max)


def pair_set(ops: ContrastiveOps, uids=None) -> set[frozenset[str]]:
    return {frozenset((p.a, p.b)) for p in ops.boundary_pairs(uids)}


def S(*exemplars) -> Subset:
    return Subset.of(e.uid for e in exemplars)


# ---- boundary pairs ---------------------------------------------------------------------------------


def test_adjacent_boundary_pairs() -> None:
    assert pair_set(make_ops()) == {
        frozenset((A.uid, B.uid)),
        frozenset((B.uid, C.uid)),
        frozenset((D.uid, E.uid)),
    }


def test_non_adjacent_pairs_only_when_configured() -> None:
    pairs = pair_set(make_ops(cfg=ContrastiveConfig(adjacent_only=False)))
    assert frozenset((A.uid, C.uid)) in pairs
    assert len(pairs) == 4
    distances = {p.distance for p in make_ops(cfg=ContrastiveConfig(adjacent_only=False)).boundary_pairs()}
    assert distances == {1, 2}


def test_tau_is_inclusive() -> None:
    x, y = ex("X", 0), ex("Y", 1)
    ops = make_ops(extra=[(x, [1.0, 0.0]), (y, [0.7, math.sqrt(1 - 0.49)])])
    assert frozenset((x.uid, y.uid)) in pair_set(ops, [x.uid, y.uid])
    assert y.uid in ops.candidates_for(x.uid)


def test_variants_of_one_item_never_pair() -> None:
    a2 = ex("A", 0, "another rationale")
    b2 = ex("B", 1, "another rationale")
    ops = make_ops(extra=[(a2, angle_vec(0)), (b2, angle_vec(10))])
    for p in ops.boundary_pairs():
        assert p.a.split("#")[0] != p.b.split("#")[0]
    assert set(ops.candidates_for(B.uid)) == {A.uid, a2.uid, C.uid}
    assert ops.candidates_for(B.uid, exclude_items={"A"}) == [C.uid]


def test_candidates_for_is_c_i() -> None:
    ops = make_ops()
    assert ops.candidates_for(A.uid) == [B.uid]
    assert set(ops.candidates_for(B.uid)) == {A.uid, C.uid}
    assert ops.candidates_for(D.uid) == [E.uid]
    assert ops.candidates_for(F.uid) == []


# ---- score ------------------------------------------------------------------------------------------


def test_coverage_score() -> None:
    ops = make_ops()
    assert ops.score(S(A, B, C, D)) == pytest.approx(3 / 4)
    assert ops.score(S(A, F)) == 0.0
    assert ops.score(S(A)) == 0.0
    assert ops.score(S(A, B, D, E)) == 1.0


def test_pair_ratio_score() -> None:
    adjacent = make_ops(cfg=ContrastiveConfig(score="pair_ratio"))
    literal = make_ops(cfg=ContrastiveConfig(score="pair_ratio", adjacent_only=False))
    assert adjacent.score(S(A, B, C, D)) == pytest.approx(2 / 6)
    assert literal.score(S(A, B, C, D)) == pytest.approx(3 / 6)


# ---- operators --------------------------------------------------------------------------------------


def test_add_inserts_boundary_neighbour() -> None:
    ops = make_ops()
    rng = np.random.default_rng(0)
    assert ops.add(S(A), rng) == S(A, B)
    grown = ops.add(S(B), rng)
    assert grown in (S(B, A), S(B, C))
    assert ops.add(S(F), rng) is None


def test_add_respects_k_max() -> None:
    ops = make_ops(k_max=2)
    assert ops.add(S(A, D), np.random.default_rng(0)) is None


def test_swap_replaces_with_boundary_neighbour() -> None:
    ops = make_ops()
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = ops.swap(S(A, D), rng)
        assert out in (S(B, D), S(A, E))


def test_swap_and_flip_never_remove_experts() -> None:
    a_exp = ex("A", 0, "expert rationale", expert=True)
    ops = make_ops(extra=[(a_exp, angle_vec(0))])
    rng = np.random.default_rng(1)
    for _ in range(30):
        swapped = ops.swap(S(a_exp, D), rng)
        assert swapped is not None and a_exp.uid in swapped
        flipped = ops.flip(S(a_exp, D), rng)
        assert flipped is not None and a_exp.uid in flipped
    assert ops.flip(S(a_exp), rng) is None
    assert ops.swap(S(a_exp), rng) is None


def test_flip_uses_unused_items() -> None:
    ops = make_ops()
    rng = np.random.default_rng(2)
    for _ in range(30):
        out = ops.flip(S(A, B), rng)
        assert out is not None and len(out) == 2
        assert len({u.split("#")[0] for u in out}) == 2


def test_operators_never_put_two_variants_of_an_item_in_a_subset() -> None:
    pool, index = dense_pool(n_items=30, variants=3)
    ops = ContrastiveOps(pool, index, INT4, ContrastiveConfig(), 4, 16)
    rng = np.random.default_rng(3)
    current = pool.random_subset(rng, 4, 8)
    for step in range(300):
        op = (ops.add, ops.swap, ops.flip)[step % 3]
        nxt = op(current, rng)
        if nxt is not None:
            assert pool.is_valid(nxt, 4, 16), "operator produced an invalid subset"
            current = nxt


# ---- candidate generation ---------------------------------------------------------------------------


@pytest.fixture
def dense_ops() -> ContrastiveOps:
    pool, index = dense_pool()
    return ContrastiveOps(pool, index, INT4, ContrastiveConfig(), 4, 16)


def _counts(cands) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cands:
        out[c.op] = out.get(c.op, 0) + 1
    return out


def test_generate_exactly_256_unique_valid_with_quotas(dense_ops: ContrastiveOps) -> None:
    rng = np.random.default_rng(0)
    incumbent = dense_ops.pool.random_subset(rng, 8, 8)
    expert = dense_ops.pool.random_subset(rng, 4, 4)
    cands = dense_ops.generate_candidates(incumbent, expert, evaluated=set(), rng=rng)
    assert len(cands) == 256
    assert len({c.subset.key() for c in cands}) == 256
    assert all(dense_ops.pool.is_valid(c.subset, 4, 16) for c in cands)
    assert _counts(cands) == {"expert": 1, "add": 64, "swap": 64, "flip": 64, "random": 63}


def test_generate_excludes_evaluated_and_is_deterministic(dense_ops: ContrastiveOps) -> None:
    incumbent = dense_ops.pool.random_subset(np.random.default_rng(1), 8, 8)
    first = dense_ops.generate_candidates(incumbent, None, set(), np.random.default_rng(5))
    again = dense_ops.generate_candidates(incumbent, None, set(), np.random.default_rng(5))
    assert [c.subset for c in first] == [c.subset for c in again]
    assert _counts(first)["random"] == 64

    seen = {c.subset.key() for c in first} | {incumbent.key()}
    second = dense_ops.generate_candidates(incumbent, None, seen, np.random.default_rng(6))
    assert not {c.subset.key() for c in second} & seen


def test_generate_backfills_when_add_impossible(dense_ops: ContrastiveOps) -> None:
    rng = np.random.default_rng(2)
    full = dense_ops.pool.random_subset(rng, 16, 16)
    cands = dense_ops.generate_candidates(full, None, set(), rng)
    counts = _counts(cands)
    assert len(cands) == 256
    assert "add" not in counts
    assert counts["swap"] == 128


def test_no_contrastive_mix_has_only_flip_and_random(dense_ops: ContrastiveOps) -> None:
    rng = np.random.default_rng(4)
    incumbent = dense_ops.pool.random_subset(rng, 8, 8)
    cands = dense_ops.generate_candidates(
        incumbent, None, set(), rng, mix=OperatorMix(add=0, swap=0, flip=128)
    )
    assert _counts(cands) == {"flip": 128, "random": 128}


def test_evaluated_expert_subset_is_skipped(dense_ops: ContrastiveOps) -> None:
    rng = np.random.default_rng(7)
    incumbent = dense_ops.pool.random_subset(rng, 8, 8)
    expert = dense_ops.pool.random_subset(rng, 4, 4)
    cands = dense_ops.generate_candidates(incumbent, expert, {expert.key()}, rng)
    assert "expert" not in _counts(cands)
    assert len(cands) == 256


def test_quota_overflow_rejected(dense_ops: ContrastiveOps) -> None:
    incumbent = dense_ops.pool.random_subset(np.random.default_rng(0), 8, 8)
    with pytest.raises(ValueError, match="exceed"):
        dense_ops.generate_candidates(
            incumbent, None, set(), np.random.default_rng(0), n=10, mix=OperatorMix(5, 5, 5)
        )
