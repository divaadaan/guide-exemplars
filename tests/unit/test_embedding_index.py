from __future__ import annotations

import numpy as np
import pytest

from guide_exemplars.core.embedding_index import EmbeddingIndex


def test_rows_are_normalized_and_sim_is_cosine() -> None:
    idx = EmbeddingIndex(["a", "b", "c"], np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
    assert idx.sim("a", "a") == pytest.approx(1.0)
    assert idx.sim("a", "b") == pytest.approx(0.0)
    assert idx.sim("a", "c") == pytest.approx(np.sqrt(0.5))
    assert idx.sim("a", "c") == idx.sim("c", "a")
    np.testing.assert_allclose(np.linalg.norm(idx.vectors(), axis=1), 1.0)


def test_submatrix_follows_requested_order() -> None:
    idx = EmbeddingIndex(["a", "b", "c"], np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    sub = idx.sim_matrix(["c", "a"])
    assert sub.shape == (2, 2)
    assert sub[0, 1] == pytest.approx(idx.sim("c", "a"))


def test_zero_vector_does_not_produce_nan() -> None:
    idx = EmbeddingIndex(["a", "z"], np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert idx.sim("a", "z") == 0.0


def test_validation() -> None:
    with pytest.raises(ValueError, match="expected"):
        EmbeddingIndex(["a"], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="duplicate"):
        EmbeddingIndex(["a", "a"], np.ones((2, 3)))
    idx = EmbeddingIndex(["a"], np.ones((1, 3)))
    with pytest.raises(KeyError):
        idx.sim("a", "missing")
    assert "a" in idx and len(idx) == 1 and idx.dim == 3
