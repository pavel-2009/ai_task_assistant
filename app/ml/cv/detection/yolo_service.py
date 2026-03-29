"""
Сервис для работы с моделью YOLOv8.
"""

import asyncio
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.core import config


MODEL_PATH = Path(__file__).with_name("yolov8n.pt")
VISUALIZATION_DIR = Path(__file__).parent.parent.parent / "avatars" / "visualizations"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class YoloService:
    """Сервис для инференса модели YOLOv8."""

    def __init__(self, model_path: Path | str = MODEL_PATH):
        """Инициализация модели."""
        self.model = YOLO(model_path)
        self.model.to("cpu")

    def predict_and_visualize(self, image_bytes: bytes, task_id: int) -> list[dict]:
        """Получения предсказаний модели для изображения"""

        img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        results = self.model.predict(
            image,
            save=False,
            verbose=False,
            conf=config.YOLO_CONF_THRESHOLD,
        )

        dict_result = []

        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidence = result.boxes.conf

            for box, cls, conf in zip(boxes, classes, confidence):
                dict_result.append({
                    "class": int(cls),
                    "class_name": self.model.names[int(cls)],
                    "confidence": float(conf),
                    "box": [float(coord) for coord in box],
                })

        im = results[0].plot()
        Image.fromarray(im).save(VISUALIZATION_DIR / f"result_{task_id}.jpg")

        return dict_result, VISUALIZATION_DIR / f"result_{task_id}.jpg"

    def predict(self, image_bytes: bytes) -> list[dict]:
        """Предсказания модели для изображения без сохранения визуализации"""

        img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        results = self.model.predict(
            image,
            save=False,
            verbose=False,
            conf=config.YOLO_CONF_THRESHOLD,
        )

        dict_result = []

        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidence = result.boxes.conf

            for box, cls, conf in zip(boxes, classes, confidence):
                dict_result.append({
                    "class": int(cls),
                    "class_name": self.model.names[int(cls)],
                    "confidence": float(conf),
                    "box": [float(coord) for coord in box],
                })

        return dict_result

    async def predict_async(self, image_bytes: bytes) -> list[dict]:
        """Асинхронные предсказания модели для изображения"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, image_bytes)

    def export_onnx(self):
        """Экспорт модели в ONNX"""

        onnx_path = self.model.export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            half=False,
            simplify=True,
            opset=12,
            project="checkpoints/onnx",
            name="best_onnx",
        )
        return onnx_path


if __name__ == "__main__":
    service = YoloService(MODEL_PATH)
    onnx_path = service.export_onnx()
    print(f"Успешно экспортированно в {onnx_path}")
