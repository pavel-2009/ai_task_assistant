"""
ONNX-сервис для YOLOv8 с возвратом имени класса
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import cv2
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ONNX_WEIGHTS_PATH = 'yolov8n.onnx'
VISUALIZATION_DIR = Path(__file__).parent.parent.parent / 'avatars' / 'visualizations'

# COCO class names
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

class YoloONNXService:
    """ONNX-сервис для инференса YOLOv8 с возвратом имени класса"""

    def __init__(self, model_path: str = ONNX_WEIGHTS_PATH):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def letterbox(self, image: np.ndarray, new_size=640):
        """Letterbox resize с сохранением пропорций"""
        h, w = image.shape[:2]
        scale = min(new_size / w, new_size / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
        top = (new_size - nh) // 2
        left = (new_size - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        return canvas, scale, left, top

    def preprocess(self, image_bytes: bytes):
        """Преобразование изображения для модели"""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)  # PIL → NumPy
        img, scale, pad_x, pad_y = self.letterbox(img_np)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return img, scale, pad_x, pad_y, img_np.shape[:2]

    def postprocess(self, output: list[np.ndarray], scale: float, pad_x: int, pad_y: int, orig_shape: tuple[int, int], conf_threshold: float = 0.45, iou_threshold: float = 0.55):
        """Постпроцессинг выводов ONNX"""
        conf_threshold = float(conf_threshold)
        iou_threshold = float(iou_threshold)

        preds = output[0].squeeze(0).T  # (8400, 84)
        boxes = preds[:, :4]  # x_center, y_center, w, h (canvas 640)
        class_scores = preds[:, 4:]

        # 🔹 Без сигмоида, просто выбираем максимальный класс
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        mask = confidences > conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Конвертация в [x1, y1, x2, y2] + обратное масштабирование
        x_c = (boxes[:,0] - pad_x) / scale
        y_c = (boxes[:,1] - pad_y) / scale
        w = boxes[:,2] / scale
        h = boxes[:,3] / scale

        x1 = x_c - w/2
        y1 = y_c - h/2
        x2 = x_c + w/2
        y2 = y_c + h/2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[:,2] = boxes_xyxy[:,2] - boxes_xyxy[:,0]
        boxes_xywh[:,3] = boxes_xyxy[:,3] - boxes_xyxy[:,1]

        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), confidences.tolist(), conf_threshold, iou_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w, h = boxes_xywh[i]
                class_id = int(class_ids[i])
                results.append({
                    "class_name": COCO_CLASSES[class_id],
                    "confidence": float(confidences[i]),
                    "box": [float(x1), float(y1), float(x1+w), float(y1+h)]
                })
        return results

    def predict(self, image_bytes: bytes):
        input_tensor, scale, pad_x, pad_y, orig_shape = self.preprocess(image_bytes)
        output = self.session.run(None, {self.input_name: input_tensor})
        results = self.postprocess(
            output, scale, pad_x, pad_y, orig_shape,
            conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", 0.35)),
            iou_threshold=float(os.getenv("YOLO_IOU_THRESHOLD", 0.55))
        )
        return results

    def predict_and_visualize(self, image_bytes: bytes, task_id: int):
        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        results = self.predict(image_bytes)
        for det in results:
            x1, y1, x2, y2 = map(int, det["box"])
            cls_name = det["class_name"]
            conf = det["confidence"]
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        output_path = VISUALIZATION_DIR / f"result_{task_id}_onnx.jpg"
        cv2.imwrite(str(output_path), img)
        return results, output_path