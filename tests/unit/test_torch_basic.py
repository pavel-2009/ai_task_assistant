import pytest
import torch

from app.utils.torch_basic import linear_regression_training


def test_linear_regression_y_eq_2x():
    x = torch.arange(1, 6, dtype=torch.float32)
    y = 2 * x
    w, b = linear_regression_training(x, y)
    assert w == pytest.approx(2.0, abs=0.1)
    assert b == pytest.approx(0.0, abs=0.2)


def test_linear_regression_y_eq_2x_plus_1():
    x = torch.arange(1, 6, dtype=torch.float32)
    y = 2 * x + 1
    w, b = linear_regression_training(x, y)
    assert w == pytest.approx(2.0, abs=0.1)
    assert b == pytest.approx(1.0, abs=0.2)


def test_linear_regression_negative_slope():
    x = torch.arange(1, 6, dtype=torch.float32)
    y = -3 * x + 2
    w, b = linear_regression_training(x, y)
    assert w == pytest.approx(-3.0, abs=0.1)
    assert b == pytest.approx(2.0, abs=0.2)
