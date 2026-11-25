from __future__ import annotations

import math

import numpy as np
import pytest
from _factories import INT4, build, dense_pool, ex, obs
from scipy.stats import norm

from guide_exemplars.core.bo import BayesianSelector, expected_improvement, select_final
from guide_exemplars.core.contrastive import Candidate
from guide_exemplars.core.featurize import SetEmbeddingFeaturizer
from guide_exemplars.core.objective import weights_for
from guide_exemplars.core.types import Subset


def test_expected_improvement_closed_form() -> None:
    ei = expected_improvement(np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.0, 2.0]), best=0.0)
    assert ei[0] == pytest.approx(norm.cdf(1.0) + norm.pdf(1.0))
    assert ei[1] == 0.0
    z = 0.25
    assert ei[2] == pytest.approx(0.5 * norm.cdf(z) + 2.0 * norm.pdf(z))
    assert (expected_improvement(np.array([-5.0]), np.array([1e-3]), best=0.0) >= 0).all()


def test_xi_reduces_ei() -> None:
    base = expected_improvement(np.array([0.2]), np.array([0.3]), 0.0, xi=0.0)
    explore = expected_improvement(np.array([0.2]), np.array([0.3]), 0.0, xi=0.1)
    assert explore[0] < base[0]


def test_suggest_with_too_few_observations_is_random() -> None:
    sel = BayesianSelector(SetEmbeddingFeaturizer(), random_state=0)
    cands = [Candidate(Subset.of([f"u{i}"]), "random") for i in range(5)]
    pick, ei = sel.suggest(cands, weights_for(0.5), np.random.default_rng(0))
    assert pick in cands and math.isnan(ei)
    with pytest.raises(ValueError):
        sel.suggest([], weights_for(0.5), np.random.default_rng(0))


def test_rescalarization_follows_weights_and_acc_star() -> None:
    sel = BayesianSelector(SetEmbeddingFeaturizer(), random_state=0)
    sel.observe(obs(Subset.of(["a", "b"]), 6, contrastive=0.5))
    sel.observe(obs(Subset.of(["a", "b", "c", "d"]), 8, contrastive=0.0))
    assert sel.acc_star() == pytest.approx(0.8)
    # w1=0.5 → w2=0.4, w3=0.1: obs1 = max(-0.1, -0.8) + 0.05 ; obs2 = max(0, -1.6) + 0
    assert sel.scalarize(weights_for(0.5)) == pytest.approx([-0.05, 0.0])
    # w1=0.25 → w2=0.6, w3=0.15: obs1 = max(-0.05, -1.2) + 0.075
    assert sel.scalarize(weights_for(0.25)) == pytest.approx([0.025, 0.0])
    # a new, better observation moves Acc* and therefore every earlier target
    sel.observe(obs(Subset.of(["e", "f"]), 9))
    assert sel.scalarize(weights_for(0.5))[1] == pytest.approx(0.5 * (0.8 - 0.9))


def _ranked_pool():
    """40 items, 10 per label; embeddings cluster by label so the featurizer can see the signal."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 6))
    spec = []
    for i in range(40):
        label = i % 4
        spec.append((ex(f"i{i}", label), list(centres[label] + rng.normal(scale=0.1, size=6))))
    return build(spec)


def test_gp_prefers_region_with_high_observed_objective() -> None:
    pool, index = _ranked_pool()
    feat = SetEmbeddingFeaturizer(n_components=4)
    feat.fit(pool, index, INT4, k_max=8)
    by_label = {lbl: [e.uid for e in pool if e.label == lbl] for lbl in "0123"}
    sel = BayesianSelector(feat, random_state=0)
    rng = np.random.default_rng(1)
    # Synthetic truth: accuracy rises with the mean label rank of the demonstrations.
    for k in range(14):
        s = pool.random_subset(rng, 4, 4)
        mean_rank = np.mean([INT4.rank(pool.get(u).label) for u in s])
        sel.observe(obs(s, n_correct=round(100 * mean_rank / 3), n_total=100, index=k))
    high = Candidate(Subset.of(by_label["3"][:4]), "add")
    low = Candidate(Subset.of(by_label["0"][:4]), "add")
    pick, ei = sel.suggest([low, high], weights_for(0.5), np.random.default_rng(0))
    assert pick is high
    assert ei > 0


def test_suggest_is_deterministic_given_seeds() -> None:
    pool, index = dense_pool(n_items=30)
    feat = SetEmbeddingFeaturizer()
    feat.fit(pool, index, INT4, k_max=8)

    def run():
        sel = BayesianSelector(feat, random_state=3)
        r = np.random.default_rng(0)
        for k in range(6):
            sel.observe(obs(pool.random_subset(r, 4, 8), n_correct=k + 2, index=k))
        cands = [Candidate(pool.random_subset(r, 4, 8), "random") for _ in range(20)]
        return sel.suggest(cands, weights_for(0.6), np.random.default_rng(9))

    a, b = run(), run()
    assert a[0].subset == b[0].subset and a[1] == b[1]


def test_select_final_lexicographic() -> None:
    big_contrastive = obs(Subset.of(["a", "b", "c"]), 8, contrastive=0.9, index=0)
    small_contrastive = obs(Subset.of(["d", "e"]), 8, contrastive=0.5, index=1)
    lower_acc = obs(Subset.of(["f"]), 7, contrastive=1.0, index=2)
    assert select_final([lower_acc, small_contrastive, big_contrastive]) is big_contrastive

    same_c_small = obs(Subset.of(["g", "h"]), 8, contrastive=0.9, index=3)
    assert select_final([big_contrastive, same_c_small]) is same_c_small


def test_select_final_compares_exact_accuracy_across_val_sizes() -> None:
    a = obs(Subset.of(["a"]), 2, n_total=3)
    b = obs(Subset.of(["b"]), 6667, n_total=10000)
    assert select_final([a, b]) is b


def test_select_final_tie_broken_by_key_deterministically() -> None:
    x = obs(Subset.of(["x1", "x2"]), 5)
    y = obs(Subset.of(["y1", "y2"]), 5)
    assert select_final([x, y]) is select_final([y, x])
    with pytest.raises(ValueError):
        select_final([])
