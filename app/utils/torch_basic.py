"""
Базовые операции с PyTorch
"""

import torch

import typing


def grad_study():
    """Пример функции для изучения градиентов в PyTorch"""
    x = torch.tensor([5.0, 5.0, 5.0], requires_grad=True)

    y = x ** 2 + 3 * x + 1

    y.backward(torch.ones_like(x))

    print("x:", x)
    print("x.grad:", x.grad)


def linear_regression_training(X: torch.Tensor, Y: torch.Tensor) -> typing.Tuple[float, float]:
    """Пример обучения простой линейной регрессии в PyTorch"""

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    for epoch in range(10000):
        y_pred = X * w + b
        loss = torch.mean((y_pred - Y) ** 2)

        loss.backward()

        w.data = w.data - 0.01 * w.grad.data
        b.data = b.data - 0.01 * b.grad.data

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, w: {w.item()}, b: {b.item()}")

        w.grad.data.zero_()
        b.grad.data.zero_()

    return w.item(), b.item()


if __name__ == "__main__":
    grad_study()
    linear_regression_training()
