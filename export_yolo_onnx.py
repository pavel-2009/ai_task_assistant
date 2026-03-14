"""
Скрипт для экспорта модели YOLOv8 в формат ONNX
"""

from app.ml.yolo_service import YoloService
from pathlib import Path


def export_yolo_to_onnx():
    """Экспортирует модель YOLOv8 в ONNX формат"""
    
    print("🚀 Начинаю экспорт YOLOv8 в ONNX...")
    
    # Инициализируем сервис с PyTorch моделью
    service = YoloService(model_path='app/ml/yolov8n.pt')
    
    # Экспортируем в ONNX
    onnx_path = service.export_onnx()
    
    print(f"✅ Успешно экспортирована модель в ONNX")
    print(f"📁 Путь: {onnx_path}")
    
    # Проверяем, что файл существует
    if Path(onnx_path).exists():
        file_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"📊 Размер файла: {file_size:.2f} МБ")
    
    return onnx_path


if __name__ == '__main__':
    export_yolo_to_onnx()
