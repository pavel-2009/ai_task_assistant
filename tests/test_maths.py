"""
Тестирования математических операций
"""

import numpy as np

from app.utils import math_ops


def test_equal_add():
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([1, 2, 3])

    assert math_ops.cosine_similarity(vec1, vec2) == 1.0


def test_orthogonal():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])

    assert math_ops.cosine_similarity(vec1, vec2) == 0.0


def test_opposite():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([-1, 0, 0])

    assert math_ops.cosine_similarity(vec1, vec2) == -1.0