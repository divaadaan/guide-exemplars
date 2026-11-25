from __future__ import annotations

import numpy as np
import pytest

from guide_exemplars.benchmark.bootstrap import bootstrap, paired_delta
from guide_exemplars.core.pool import Pool
from guide_exemplars.core.selectors.base import SelectionContext, StaticPolicy
from guide_exemplars.core.selectors.baselines import (
    KNNPolicy,
    KNNSelector,
    NaiveSelector,
    RandomSelector,
    VoteKSelector,
)
from guide_exemplars.core.types import Exemplar
from guide_exemplars.llm.embedders import HashEmbedder
from support.toy import toy_items

# ---- bootstrap --------------------------------------------------------------------------------------


def test_identical_predictions_have_zero_delta() -> None:
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 4, 80).tolist()
    pred = [g if rng.random() < 0.7 else min(3, g + 1) for g in gold]
    d = paired_delta(gold, [pred], [pred], 4, B=200)
    for m in ("acc", "qwk", "adj_err"):
        assert d[m]["delta"] == 0.0 and d[m]["lo"] == 0.0 and d[m]["hi"] == 0.0


def test_ci_contains_point_and_shrinks_with_n() -> None:
    rng = np.random.default_rng(1)

    def width(n: int) -> float:
        gold = rng.integers(0, 3, n).tolist()
        pred = [g if rng.random() < 0.6 else (g + 1) % 3 for g in gold]
        r = bootstrap(gold, [pred], 3, B=300)
        assert r["acc"]["lo"] <= r["acc"]["point"] <= r["acc"]["hi"]
        return r["acc"]["hi"] - r["acc"]["lo"]

    assert width(400) < width(40)


def test_bootstrap_coverage_is_roughly_nominal() -> None:
    rng = np.random.default_rng(2)
    p_true, n, hits, trials = 0.7, 120, 0, 60
    for t in range(trials):
        gold = [1] * n
        pred = [1 if rng.random() < p_true else 0 for _ in range(n)]
        r = bootstrap(gold, [pred], 2, B=200, seed=t)
        hits += r["acc"]["lo"] <= p_true <= r["acc"]["hi"]
    assert hits / trials >= 0.85


def test_multiple_draws_are_averaged() -> None:
    gold = [0, 1, 2, 1]
    r = bootstrap(gold, [[0, 1, 2, 1], [0, 0, 0, 0]], 3, B=50)
    assert r["acc"]["point"] == pytest.approx((1.0 + 0.25) / 2)


def test_failures_and_positive_delta() -> None:
    gold = [1] * 50
    good = [1] * 45 + [None] * 5
    bad = [1] * 25 + [0] * 25
    d = paired_delta(gold, [good], [bad], 2, B=300)
    assert d["acc"]["delta"] == pytest.approx(0.4)
    assert d["acc"]["p_gt0"] > 0.99


# ---- baselines --------------------------------------------------------------------------------------

ITEMS = toy_items(n_per_label=10)
POOL = Pool(Exemplar.from_item(it, f"why {it.item_id}") for it in ITEMS)


def ctx(k: int = 8, seed: int = 0) -> SelectionContext:
    return SelectionContext(pool=POOL, k=k, rng=np.random.default_rng(seed), embedder=HashEmbedder(128))


async def test_naive_is_label_balanced_and_seed_independent() -> None:
    a = await NaiveSelector().fit(ctx(seed=0))
    b = await NaiveSelector().fit(ctx(seed=99))
    assert a.exemplars == b.exemplars
    labels = [e.label for e in a.exemplars]
    assert len(a.exemplars) == 8 and all(labels.count(lbl) == 2 for lbl in set(labels))


async def test_random_is_seeded_and_distinct() -> None:
    a = await RandomSelector().fit(ctx(seed=1))
    b = await RandomSelector().fit(ctx(seed=1))
    c = await RandomSelector().fit(ctx(seed=2))
    assert a.exemplars == b.exemplars != c.exemplars
    assert len({e.item_id for e in a.exemplars}) == 8


async def test_vote_k_is_diverse() -> None:
    vk = await VoteKSelector().fit(ctx(k=6))
    rnd = await RandomSelector().fit(ctx(k=6, seed=3))
    emb = HashEmbedder(128)

    async def mean_pairwise_sim(policy: StaticPolicy) -> float:
        v = await emb.embed([e.text for e in policy.exemplars])
        s = v @ v.T
        return float((s.sum() - np.trace(s)) / (len(v) * (len(v) - 1)))

    assert len({e.item_id for e in vk.exemplars}) == 6
    assert await mean_pairwise_sim(vk) < await mean_pairwise_sim(rnd)


async def test_knn_returns_nearest_other_items() -> None:
    policy = await KNNSelector().fit(ctx(k=3))
    assert isinstance(policy, KNNPolicy) and policy.kind == "per_query"
    query = ITEMS[0]
    with pytest.raises(KeyError):
        policy.demos_for(query)
    await policy.prepare([query])
    demos = policy.demos_for(query)
    assert len(demos) == 3 and all(d.item_id != query.item_id for d in demos)
    q = (await HashEmbedder(128).embed([query.text]))[0]
    sims = {d.item_id: float(policy.vectors[i] @ q) for i, d in enumerate(policy.exemplars)}
    others = sorted((s for iid, s in sims.items() if iid != query.item_id), reverse=True)
    assert [sims[d.item_id] for d in demos] == pytest.approx(others[:3])
