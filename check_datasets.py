"""
Временный скрипт для проверки датасетов на наличие ошибок и несоответствий.
"""

from torchvision import transforms
import torchvision.transforms.functional as TF

from app.ml.config import config
from app.ml.datasets import TaskImageDataset


if __name__ == "__main__":
    dataset = TaskImageDataset(
        root_dir=config.data_dir,
        transforms=[
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )
    print(f"Количество изображений в датасете: {len(dataset)}")

    # Проверка первого элемента датасета
    image, label = dataset[0]
    print(f"Первое изображение: {label}, размер: {image.shape}")

    # Проверка всех изображений на наличие ошибок
    for i in range(len(dataset)):
        try:
            image, label = dataset[i]
        except Exception as e:
            print(f"Ошибка при загрузке изображения {i}: {e}")

    # Применение трансформаций к изображениям
    for i in range(len(dataset)):
        try:
            image, label = dataset[i]
            filename = f"image_{i}.png" 
            print(f"Трансформации успешно применены к изображению {filename}")

            image = TF.to_pil_image(image)
            image.save(f"transformed_{filename}")
        except Exception as e:
            print(f"Ошибка при применении трансформаций к изображению {filename}: {e}")


    from app.ml.models_nn import SimpleCNN
    from app.ml.config import config

    # Создай модель
    model = SimpleCNN(num_classes=config.num_classes)
    print(f"Модель создана: {model}")

    # Возьми один батч из датасета (через DataLoader)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    images, labels = next(iter(loader))

    print(f"Входной тензор: {images.shape}")

    # Прогони через модель
    try:
        output = model(images)
        print(f"Выходной тензор (логиты): {output.shape}")
        print("✅ Модель работает! Размеры сходятся.")
    except RuntimeError as e:
        print(f"❌ Ошибка размеров: {e}")