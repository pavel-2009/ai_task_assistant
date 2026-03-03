"""
Модуль конфигурации для машинного обучения
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class MLConfig(BaseSettings):
    """
    Базовая конфигурация для машинного обучения
    """

    data_dir: Path = Path(__file__).parent.parent.parent / "avatars"
    output_dir: Path = Path(__file__).parent.parent.parent / "checkpoints"
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    num_classes: int = 3
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]
    learning_rate: float = 0.001
    num_epochs: int = 10


config = MLConfig()