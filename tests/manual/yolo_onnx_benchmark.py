"""
Ручной бенчмарк/сравнение PyTorch YOLO vs ONNX YOLO.
Запуск: python tests/manual/yolo_onnx_benchmark.py
"""

from pathlib import Path
import io
import time

from PIL import Image

from app.ml.cv.detection.yolo_service import YoloService
from app.ml.cv.detection.yolo_onnx_service import YoloONNXService


def iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _build_test_image_bytes() -> bytes:
    image = Image.new("RGB", (640, 640), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def main() -> None:
    base_dir = Path("app/ml/cv/detection")
    pytorch_model = base_dir / "yolov8n.pt"
    onnx_model = base_dir / "yolov8n.onnx"

    image_bytes = _build_test_image_bytes()

    yolo_service = YoloService(str(pytorch_model))
    yolo_onnx_service = YoloONNXService(str(onnx_model))

    py_results, _ = yolo_service.predict_and_visualize(image_bytes, task_id=0)
    onnx_results, _ = yolo_onnx_service.predict_and_visualize(image_bytes, task_id=0)

    start = time.time()
    for _ in range(3):
        py_results, _ = yolo_service.predict_and_visualize(image_bytes, task_id=0)
    py_time = time.time() - start

    start = time.time()
    for _ in range(3):
        onnx_results, _ = yolo_onnx_service.predict_and_visualize(image_bytes, task_id=0)
    onnx_time = time.time() - start

    print("=== PyTorch YOLO ===")
    print(f"objects: {len(py_results)}")
    print(f"time: {py_time:.3f}s")

    print("=== ONNX YOLO ===")
    print(f"objects: {len(onnx_results)}")
    print(f"time: {onnx_time:.3f}s")

    used: set[int] = set()
    for i, py_obj in enumerate(py_results):
        best_iou = 0.0
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
        print(f"obj {i + 1}: iou={best_iou:.3f}")

    speedup = py_time / onnx_time if onnx_time > 0 else float("inf")
    print(f"ONNX speedup x{speedup:.2f}")


if __name__ == "__main__":
    main()
