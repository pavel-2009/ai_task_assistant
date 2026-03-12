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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model: nn.Module = get_pretrained_model()
        self.model.load_state_dict(torch.load(self.checkpoints_path))
        self.model.to(self.device)
        self.model.eval()

        self.val_transforms: transforms.Compose = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])


    def predict(self, image_bytes: bytes) -> dict:
        """Предсказание класса для входного изображения"""

        # На всякий пожарный - переводи модель в режим оценки
        self.model.eval()

        try:
            image = Image.open(io.BytesIO(image_bytes))

        except Exception as e:
            raise ValueError(f"Невалидное изображение: {e}")

        image_transformed = self.val_transforms(image).unsqueeze(0)
        image_transformed = image_transformed.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_transformed)

            softmax = nn.Softmax(dim=1)
            
            probabilities = softmax(outputs)
            prob, idx = torch.max(probabilities, dim=1)
            name = self.idx_to_class[idx.item()]

        return {"class_id": int(idx.item()), "class_name": name, "confidence": float(prob.item())}
