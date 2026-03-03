"""
Модуль с моделями датасетов для машинного обучения.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image

from pathlib import Path
from typing import Tuple, List


class TaskImageDataset(Dataset):
    """
    Датасет для аватарок задач приложения AI Task Assistant.
    """

    def __init__(self, root_dir: str, transforms: List = None):
        self.root_dir = root_dir
        self.image_paths = list(Path(root_dir).glob("*.jpeg*"))
        if not self.image_paths:
            raise ValueError(f"Нет изображений в директории {root_dir}")
        self.transforms = transforms
        self.classes = ["unknown"] 
        self.class_to_idx = {"unknown": 0}


    def __len__(self) -> int:
        """Количество изображений в датасете"""
        return len(self.image_paths)

    
    def __getitem__(self, index) -> Tuple[torch.Tensor, float]:
        """Получение элемента датасета по индексу. Возвращает пару (изображение, имя файла)"""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return (image, 0)
