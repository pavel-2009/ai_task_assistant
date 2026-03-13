import time
from yolo_service import YoloService
from yolo_onnx_service import YoloONNXService

# Путь к моделям и изображению
PYTORCH_MODEL = "yolov8n.pt"
ONNX_MODEL = "yolov8n.onnx"
IMAGE_PATH = "0_0.jpg"

# Загружаем изображение в байтах
with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()


yolo_service = YoloService(PYTORCH_MODEL)


# Прогрев
py_results, _  = yolo_service.predict_and_visualize(image_bytes, task_id=0)

start = time.time()


for _ in range(100 - 1):
    py_results, _ = yolo_service.predict_and_visualize(image_bytes, task_id=0)
py_time = time.time() - start

print("=== PyTorch YOLOv8 ===")
print(f"Найдено объектов: {len(py_results)}")
print(f"Время инференса: {py_time:.3f} сек")



onnx_service = YoloONNXService(ONNX_MODEL)

# Прогрев
onnx_results = onnx_service.predict(image_bytes)

start = time.time()


for _ in range(100 - 1):
    onnx_results = onnx_service.predict(image_bytes)
onnx_time = time.time() - start

print("\n=== ONNX YOLOv8 ===")
print(f"Найдено объектов: {len(onnx_results)}")
print(f"Время инференса: {onnx_time:.3f} сек")



print("\n=== Сравнение PyTorch vs ONNX ===")
for i, (py_obj, onnx_obj) in enumerate(zip(py_results, onnx_results)):
    print(f"Объект {i+1}:")
    print(f"  PyTorch: {py_obj['class_name']} {py_obj['confidence']:.3f}")
    print(f"  ONNX   : Class {onnx_obj['class']} {onnx_obj['confidence']:.3f}")
    print(f"  bbox diff: {[round(a-b,2) for a,b in zip(py_obj['box'], onnx_obj['box'])]}")


speedup = py_time / onnx_time if onnx_time > 0 else float('inf')
print(f"\nONNX быстрее PyTorch примерно в {speedup:.2f} раза")