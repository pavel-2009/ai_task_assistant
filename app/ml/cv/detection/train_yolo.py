"""
Скрипт обучения модели YOLO на кастомном датасете
"""

import ultralytics
from pathlib import Path


def train_yolo():
    """Функция обучения модели YOLO"""
    
    print("Начинаем обучение модели YOLO...")
    
    model = ultralytics.YOLO(Path(__file__).with_name('yolov8n.pt'))
    
    data_yaml_path = Path(__file__).parent.parent.parent / "data" / "data.yaml"
    
    results = model.train(
        data=str(data_yaml_path),
        epochs=50,
        imgsz=640,
        project='runs/detect',
        name='task_detector_v1'
    )
    
    print("Обучение завершено. Результаты сохранены в папке 'runs/detect/task_detector_v1'.")
    return results


if __name__ == "__main__":
    train_yolo()