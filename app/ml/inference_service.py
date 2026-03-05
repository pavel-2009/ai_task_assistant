"""
Сервис для инференса модели машинного обучения
"""

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from pathlib import Path
import io

from .models_nn import get_pretrained_model
from .config import config


class InferenceService:
    """Сервис для инференса модели машинного обучения"""

    def __init__(self, checkpoints_path: Path, idx_to_class: dict[int, str]):
        self.checkpoints_path = checkpoints_path
        self.idx_to_class = idx_to_class
        
        self.model: nn.Module = get_pretrained_model()
        self.model.load_state_dict(torch.load(self.checkpoints_path))

        self.model.eval()

        self.val_transforms: transforms.Compose = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])


    def predict(self, image_bytes: bytes) -> dict:
        """Предсказание класса для входного изображения"""

        image = Image.open(io.BytesIO(image_bytes))

        image_transformed = self.val_transforms(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_transformed)

            softmax = nn.Softmax(dim=1)
            
            probabilities = softmax(outputs)
            prob, idx = torch.max(probabilities, dim=1)
            name = self.idx_to_class[idx.item()]

        return {"class_id": idx, "class_name": name, "confidence": float(prob)}
