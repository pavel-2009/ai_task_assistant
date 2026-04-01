"""
Сервис получения эмбеддингов изображений для задач. 
Этот сервис будет использоваться для получения векторных 
представлений изображений, которые затем можно будет 
использовать для поиска похожих задач на основе визуального контента.
"""

import torch
from torchvision import transforms, models

from PIL import Image
from io import BytesIO

import numpy as np


class ImageEmbeddingService:
    """Сервис для получения эмбеддингов изображений."""
    
    def __init__(self):
        self.dimension = 512
        # Загружаем предобученную модель ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Удаляем только классификационный слой, оставляя GAP-представление размерности 2048
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.eval()  # Устанавливаем модель в режим оценки
        
        # Определяем трансформации для входных изображений
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
    def get_embedding(self, image: bytes) -> np.ndarray:
        """Получаем эмбеддинг для изображения."""
        # Декодируем изображение из байтов
        
        img = Image.open(BytesIO(image)).convert('RGB')
        
        # Применяем трансформации
        input_tensor = self.transform(img).unsqueeze(0)  # Добавляем размерность для батча
        
        with torch.no_grad():
            embedding_2048 = self.model(input_tensor).flatten().numpy().astype(np.float32)

        # Приводим к фиксированной размерности 512 для совместимости с RecSys (384 + 512 = 896)
        if embedding_2048.shape[0] >= self.dimension:
            return embedding_2048[: self.dimension]

        return np.pad(
            embedding_2048,
            (0, self.dimension - embedding_2048.shape[0]),
            mode="constant",
            constant_values=0.0,
        )
