"""
Простой пайплайн для нейронных сетей.
"""

from torch import nn
import torch


class SimpleCNN(nn.Module):

    def __init__(self, num_classes: int = 0):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 56 * 56, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x
    