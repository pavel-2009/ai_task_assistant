"""
ONNX-сервис для YOLOv8 с возвратом имени класса
"""

import onnxruntime as ort
import numpy as np
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

sess_options = ort.SessionOptions()

sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = os.cpu_count()
sess_options.inter_op_num_threads = 1

class YoloONNXService:
    """ONNX-сервис для инференса YOLOv8 с возвратом имени класса"""

    def __init__(self, model_path: str = ONNX_WEIGHTS_PATH):
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name


    def letterbox(self, image: np.ndarray, new_size=640) -> tuple[np.ndarray, float, int, int]:
        """Letterbox масштабирование с сохранением пропорций"""
        
        h, w = image.shape[:2]
        
        scale = min(new_size / w, new_size / h)
        
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))
        
        canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)
        
        top = (new_size - nh) // 2
        left = (new_size - nw) // 2
        
        canvas[top:top+nh, left:left+nw] = resized
        
        return canvas, scale, left, top


    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, float, int, int, tuple[int, int]]:
        """Преобразование изображения для модели"""
        
        img_array = np.frombuffer(image_bytes, np.uint8)
        
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        orig_shape = img.shape[:2]
        img, scale, pad_x, pad_y = self.letterbox(img)
        
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2,0,1).copy()  # HWC -> CHW
        
        input_tensor = np.expand_dims(img, axis=0)  # (1, 3, 640, 640)
        return input_tensor, scale, pad_x, pad_y, orig_shape
    

    def postprocess(self, output: list[np.ndarray], scale: float, pad_x: int, pad_y: int, orig_shape: tuple[int, int], conf_threshold: float = 0.45, iou_threshold: float = 0.55):
        """Постпроцессинг выводов ONNX"""
        conf_threshold = float(conf_threshold)
        iou_threshold = float(iou_threshold)

        preds = output[0].squeeze(0).T  # (8400, 84)
        boxes = preds[:, :4]  # x_center, y_center, w, h (canvas 640)
        class_scores = preds[:, 4:]

        # Получаем классы и уверенности
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences > conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Конвертация в [x1, y1, x2, y2] + обратное масштабирование
        x_c, y_c, w, h = boxes.T
        
        x1 = x_c - w/2
        y1 = y_c - h/2
        x2 = x_c + w/2
        y2 = y_c + h/2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        
        boxes /= scale
        
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            conf_threshold,
            iou_threshold
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                
                x1, y1, x2, y2 = boxes[i]
                
                class_id = int(class_ids[i])
                
                results.append({
                    "class_name": COCO_CLASSES[class_id],
                    "confidence": float(confidences[i]),
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })
                
        return results


    def predict(self, image_bytes: bytes) -> list[dict]:
        """Предсказание с постпроцессингом"""
        
        input_tensor, scale, pad_x, pad_y, orig_shape = self.preprocess(image_bytes)
        
        output = self.session.run(None, {self.input_name: input_tensor})
        
        results = self.postprocess(
            output, scale, pad_x, pad_y, orig_shape,
            conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", 0.35)),
            iou_threshold=float(os.getenv("YOLO_IOU_THRESHOLD", 0.55))
        )
        
        return results
    
    
    def predict_without_postprocess(self, image_bytes: bytes) -> list[np.ndarray]:
        """Предсказание без постпроцессинга (сырые данные)"""
        
        input_tensor, _, _, _, _ = self.preprocess(image_bytes)
        
        output = self.session.run(None, {self.input_name: input_tensor})
        
        return output


    def predict_and_visualize(self, image_bytes: bytes, task_id: int) -> tuple[list[dict], Path]:
        """Предсказание с визуализацией результатов"""
        
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