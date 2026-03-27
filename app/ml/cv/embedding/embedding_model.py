"""
Модель torch для получения эмбеддингов аватарок задач. Используется для рекомендаций похожих задач в системе.
"""

import torch
from torch import nn


class ImageEmbeddingModel(nn.Module):
    """Модель для получения эмбеддингов изображений задач."""
    
    def __init__(self):
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        
        self.relu = nn.ReLU()
        
        self.lin = nn.Linear(512, 512)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        x = torch.flatten(x, start_dim=1)
        x = self.lin(x)
        
        return x
        