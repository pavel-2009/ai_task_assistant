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


def iou(box1: list[float], box2: list[float]) -> float:
    """Вычисление IoU для двух боксов [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# Базовая директория проекта
BASE_DIR = Path(__file__).parent

# Пути к моделям
PYTORCH_MODEL = BASE_DIR / "yolov8n.pt"
ONNX_MODEL = BASE_DIR / "yolov8n.onnx"
IMAGE_PATH = BASE_DIR / "0_0.jpg"

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

print("\n=== Сравнение PyTorch vs ONNX ===")

used = set()

for i, py_obj in enumerate(py_results):
    
    best_iou = 0
    best_j = None
    
    for j, onnx_obj in enumerate(onnx_results):
        
        if j in used:
            continue
        
        score = iou(py_obj["box"], onnx_obj["box"])
        
        if score > best_iou:
            best_iou = score
            best_j = j
    
    if best_j is None:
        continue
    
    used.add(best_j)
    
    onnx_obj = onnx_results[best_j]
    
    print(f"\nОбъект {i+1}")
    print(f"IoU: {best_iou:.3f}")
    
    print(f"PyTorch: {py_obj['class_name']} {py_obj['confidence']:.3f}")
    print(f"ONNX   : {onnx_obj['class_name']} {onnx_obj['confidence']:.3f}")
    
    print(
        "bbox diff:",
        [round(a-b,2) for a,b in zip(py_obj["box"], onnx_obj["box"])]
    )
    
speedup = py_time / onnx_time if onnx_time > 0 else float('inf')    
    
print(f"\nONNX быстрее PyTorch примерно в {speedup:.2f} раза")