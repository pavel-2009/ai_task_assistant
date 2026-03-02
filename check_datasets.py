"""
Временный скрипт для проверки датасетов на наличие ошибок и несоответствий.
"""

from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

from app.ml.config import config
from app.ml.datasets import TaskImageDataset


if __name__ == "__main__":
    dataset = TaskImageDataset(
        root_dir=config.data_dir,
        transforms=[
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]
    )
    print(f"Количество изображений в датасете: {len(dataset)}")

    # Проверка первого элемента датасета
    image, filename = dataset[0]
    print(f"Первое изображение: {filename}, размер: {image.shape}")

    # Проверка всех изображений на наличие ошибок
    for i in range(len(dataset)):
        try:
            image, filename = dataset[i]
        except Exception as e:
            print(f"Ошибка при загрузке изображения {filename}: {e}")

    # Применение трансформаций к изображениям
    for i in range(len(dataset)):
        try:
            image, filename = dataset[i]
            print(f"Трансформации успешно применены к изображению {filename}")

            image = TF.to_pil_image(image)
            image.save(f"transformed_{filename}")
        except Exception as e:
            print(f"Ошибка при применении трансформаций к изображению {filename}: {e}")