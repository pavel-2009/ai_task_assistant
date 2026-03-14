"""
ONNX-сервис для YOLO
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import cv2

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ONNX_WEIGHTS_PATH = 'yolov8n.onnx'
VISUALIZATION_DIR = Path(__file__).parent.parent.parent / 'avatars' / 'visualizations'


class YoloONNXService:
    """Сервис ONNX для YOLO-инференса"""
    
    def __init__(self, model_path: str = ONNX_WEIGHTS_PATH):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        
        
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Препроцессинг изображения перед предсказанием"""
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((640, 640))
        
        img = np.array(image).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        return img
    
    
    def predict(self, image_bytes: bytes) -> np.ndarray:
        """Предсказания на основе изображения"""
        
        input_tensor = self.preprocess(image_bytes)
        
        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor
            }
        )
        
        return self.postprocess(outputs)


    def postprocess(self, output: np.ndarray, conf_threshold: float = os.getenv("YOLO_CONF_THRESHOLD", 0.05), iou_threshold: float = os.getenv("YOLO_IOU_THRESHOLD", 0.65)) -> list:
        """Постпроцессинг выводов модели ONNX"""
        
        preds = np.squeeze(output).T
        
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        
        conf_threshold, iou_threshold = float(conf_threshold), float(iou_threshold)
        
        mask = confidences > conf_threshold
        
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold
        )

        results = []

        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "class": int(class_ids[i]),
                    "confidence": float(confidences[i]),
                    "box": boxes_xyxy[i].tolist()
                })

        return results
    
    
    def predict_and_visualize(self, image_bytes: bytes, task_id: int):
        """Получение предсказаний и визуализация"""

        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        results = self.predict(image_bytes)

        for det in results:
            x1, y1, x2, y2 = map(int, det["box"])
            conf = det["confidence"]
            cls = det["class"]

            label = f"class {cls} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        output_path = VISUALIZATION_DIR / f"result_{task_id}_onnx.jpg"

        cv2.imwrite(str(output_path), img)

        return results, output_path