"""
Простой тест инференса модели через ONNX
"""

from yolo_onnx_service import YoloONNXService
from yolo_service import YoloService


service = YoloONNXService("yolov8n.onnx")

with open("0_0.jpg", "rb") as f:
    image_bytes = f.read()

result = service.predict(image_bytes)

print("=== ONNX ===")
print(result)


service = YoloService("yolov8n.pt")

with open("0_0.jpg", "rb") as f:
    image_bytes = f.read()

result = service.predict_and_visualize(image_bytes, 1)
print("=== YOLO ===")
print(result)