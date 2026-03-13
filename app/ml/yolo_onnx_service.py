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

load_dotenv()


class YoloONNXService:
    """Сервис ONNX для YOLO-инференса"""
    
    def __init__(self, model_path: str):
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


    def postprocess(self, output: np.ndarray, conf_threshold: float = os.getenv("YOLO_CONF_THRESHOLD", 0.45), iou_threshold: float = os.getenv("YOLO_IOU_THRESHOLD", 0.01)) -> list:
        """Постпроцессинг выводов модели ONNX"""
        
        preds = np.squeeze(output).T
        
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        
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