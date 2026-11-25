from __future__ import annotations

import numpy as np
import pytest
from _factories import INT4, dense_pool

from guide_exemplars.core.featurize import MembershipFeaturizer, SetEmbeddingFeaturizer
from guide_exemplars.core.types import Subset


def test_membership_uses_uid_coordinates_for_variants() -> None:
    pool, index = dense_pool(n_items=5, variants=2)
    feat = MembershipFeaturizer()
    feat.fit(pool, index, INT4, k_max=4)
    v0, v1 = pool.variants("i0")
    x = feat.transform([Subset.of([v0.uid]), Subset.of([v1.uid])])
    assert x.shape == (2, 10)
    assert x.sum(axis=1).tolist() == [1.0, 1.0]
    assert not np.array_equal(x[0], x[1]), "variants of one item must not share a coordinate"


def test_set_embedding_layout() -> None:
    pool, index = dense_pool(n_items=40, variants=1)
    feat = SetEmbeddingFeaturizer(n_components=4)
    feat.fit(pool, index, INT4, k_max=8)
    assert feat.dim == 4 + 4 + 1
    members = pool.uids()[:4]
    x = feat.transform([Subset.of(members)])[0]
    ranks = [INT4.rank(pool.get(u).label) for u in members]
    np.testing.assert_allclose(x[4:8], np.bincount(ranks, minlength=4) / 4)
    assert x[-1] == pytest.approx(4 / 8)


def test_set_embedding_is_order_free_and_distinguishes_sets() -> None:
    pool, index = dense_pool(n_items=40, variants=1)
    feat = SetEmbeddingFeaturizer()
    feat.fit(pool, index, INT4, k_max=16)
    uids = pool.uids()
    a = feat.transform([Subset.of(uids[:5]), Subset.of(reversed(uids[:5])), Subset.of(uids[5:10])])
    np.testing.assert_allclose(a[0], a[1])
    assert not np.allclose(a[0], a[2])


def test_pca_components_capped_by_pool_size() -> None:
    pool, index = dense_pool(n_items=5, variants=1)
    feat = SetEmbeddingFeaturizer(n_components=16)
    feat.fit(pool, index, INT4, k_max=4)
    assert feat.dim == 4 + 4 + 1


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        SetEmbeddingFeaturizer().transform([Subset.of(["x"])])
