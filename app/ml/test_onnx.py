"""
Скрипт для тестирования моделей на PyTorch и ONNX
"""

import sys
from pathlib import Path
import time

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from yolo_service import YoloService
from yolo_onnx_service import YoloONNXService

# Базовая директория проекта
BASE_DIR = Path(__file__).parent

# Пути к моделям
PYTORCH_MODEL = BASE_DIR / "yolov8n.pt"
ONNX_MODEL = BASE_DIR / "yolov8n.onnx"
IMAGE_PATH = BASE_DIR / "0_0.jpeg"

# Загружаем изображение в байтах
with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()


# --- PyTorch YOLOv8 ---
yolo_service = YoloService(str(PYTORCH_MODEL))

# Прогрев
py_results, _ = yolo_service.predict_and_visualize(image_bytes, task_id=0)

start = time.time()
for _ in range(10 - 1):
    py_results, _ = yolo_service.predict_and_visualize(image_bytes, task_id=0)
py_time = time.time() - start

print("=== PyTorch YOLOv8 ===")
print(f"Найдено объектов: {len(py_results)}")
print(f"Время инференса: {py_time:.3f} сек")


# --- ONNX YOLOv8 ---
onnx_service = YoloONNXService(str(ONNX_MODEL))

# Прогрев
onnx_results, _ = onnx_service.predict_and_visualize(image_bytes, task_id=0)

start = time.time()
for _ in range(10 - 1):
    onnx_results, _ = onnx_service.predict_and_visualize(image_bytes, task_id=0)
onnx_time = time.time() - start

print("\n=== ONNX YOLOv8 ===")
print(f"Найдено объектов: {len(onnx_results)}")
print(f"Время инференса: {onnx_time:.3f} сек")


# --- Сравнение объектов ---
print("\n=== Сравнение PyTorch vs ONNX ===")
min_len = min(len(py_results), len(onnx_results))
for i in range(min_len):
    py_obj = py_results[i]
    onnx_obj = onnx_results[i]
    print(f"Объект {i+1}:")
    print(f"  PyTorch: {py_obj['class_name']} {py_obj['confidence']:.3f}")
    print(f"  ONNX   : {onnx_obj['class_name']} {onnx_obj['confidence']:.3f}")
    print(f"  bbox diff: {[round(a-b,2) for a,b in zip(py_obj['box'], onnx_obj['box'])]}")


speedup = py_time / onnx_time if onnx_time > 0 else float('inf')
print(f"\nONNX быстрее PyTorch примерно в {speedup:.2f} раза")