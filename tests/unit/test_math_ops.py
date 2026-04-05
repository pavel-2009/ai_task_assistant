import numpy as np
import pytest

from app.utils.math_ops import cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(-1.0)


def test_cosine_similarity_different_magnitudes():
    assert cosine_similarity(np.array([1, 1]), np.array([10, 10])) == pytest.approx(1.0)


def test_cosine_similarity_negative_values():
    v1 = np.array([-1, -2, -3])
    v2 = np.array([1, 2, 3])
    assert cosine_similarity(v1, v2) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_handling():
    assert cosine_similarity(np.array([0, 0]), np.array([1, 2])) == 0.0
