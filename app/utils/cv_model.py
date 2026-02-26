"""
Модель для работы с изображениями в рамках приложения AI Task Assistant.
"""

import torch
import torchvision
import numpy as np
import cv2

from image_ops import validate_image, resize_image


model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
)
model.eval()


weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
categories = weights.meta["categories"]


def predict_image_class(image_bytes: bytes) -> str:
    """Предсказание класса изображения"""
    
    if not validate_image(image_bytes):
        raise ValueError("Invalid image")

    img_resized = resize_image(image_bytes)

    img_array = np.frombuffer(img_resized, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224)).transpose(2, 0, 1) / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std

    img_tensor = torch.from_numpy(img).float().unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return categories[predicted_class]

