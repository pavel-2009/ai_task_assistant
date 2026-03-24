"""
Модуль, содержащий функции для обучения модели.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from typing import Tuple

from ...common.config import config
from .datasets import TaskImageDataset
from .models_nn import SimpleCNN, get_pretrained_model


OUTPUT_DIR = config.output_dir


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Получить трансформации для обучения и валидации
    """

    train_transforms = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])

    validate_transforms = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])

    return train_transforms, validate_transforms


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Обучение модели на одной эпохе"""

    model.train()

    running_loss = 0.0
    num_batches = 0

    for images, labels in dataloader:
        # Перенос данных на устройство
        images, labels = images.to(device), labels.to(device)

        # Обнуление градиентов
        optimizer.zero_grad()

        # Предсказание и вычисление потерь
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратное распространение и обновление весов
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    # Возвращаем среднее значение потерь за эпоху
    return running_loss / num_batches


if __name__ == "__main__":
    # Определение устройства (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Получение трансформаций
    train_transforms, validate_transforms = get_transforms()

    # Создание датасетов и загрузчиков данных
    train_dataset = TaskImageDataset(root_dir=config.data_dir, transforms=train_transforms)
    validate_dataset = TaskImageDataset(root_dir=config.data_dir, transforms=validate_transforms)

    # Инициализация модели, функции потерь и оптимизатора
    model = get_pretrained_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Цикл обучения
    for epoch in range(config.num_epochs):

        train_loss = train_epoch(
            model,
            DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
            criterion,
            optimizer,
            device
        )
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {train_loss:.4f}")

    

    torch.save(model.state_dict(), OUTPUT_DIR / "model.pth")