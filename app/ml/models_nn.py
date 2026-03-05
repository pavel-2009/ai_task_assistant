"""
Простой пайплайн для нейронных сетей.
"""

from torch import nn
import torch
from torchvision import models

from .config import config


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
    

def get_pretrained_model() -> nn.Module:
    """Загрузка предобученной модели ResNet18"""

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.num_classes)

    return model