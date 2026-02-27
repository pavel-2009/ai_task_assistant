"""
Тестирования математических операций
"""

import numpy as np
import torch

from app.utils import math_ops
from app.utils import torch_basic


def test_equal_add():
    """Тестирование косинусного сходства для идентичных векторов"""
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([1, 2, 3])

    assert math_ops.cosine_similarity(vec1, vec2) == 1.0


def test_orthogonal():
    """Тестирование косинусного сходства для ортогональных векторов"""
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])

    assert math_ops.cosine_similarity(vec1, vec2) == 0.0


def test_opposite():
    """Тестирование косинусного сходства для противоположных векторов"""
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([-1, 0, 0])

    assert math_ops.cosine_similarity(vec1, vec2) == -1.0


def test_linreg_study():
    """Тестирование функции обучения линейной регрессии в PyTorch"""

    X = [[1], [2], [3], [4], [5]]
    Y = [[2], [4], [6], [8], [10]]
    w, b = torch_basic.linear_regression_training(X, Y)

    assert abs(w - 2) < 0.1
    assert abs(b) < 0.1