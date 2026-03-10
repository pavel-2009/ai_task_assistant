"""
Сервис для работы с моделью YOLOv8.
"""

from ultralytics import YOLO
from PIL import Image
import io

from pathlib import Path


MODEL_PATH = 'yolov8n.pt'   # Path(__file__).parent.parent.parent / 'runs' / 'detect' / 'runs' / 'detect' / 'task_detector_v1' / 'weights' / 'best.pt'


class YoloService:
    """Сервис для инференса модели YOLOv8."""
    
    def __init__(self, model_path: Path | str = MODEL_PATH):
        """Инициализация модели."""
        self.model = YOLO(model_path)
        self.model.to('cpu')
        
        
    def predict(self, image_bytes: bytes) -> list[dict]:
        """Получения предсказаний модели для изображения"""
        
        image = Image.open(io.BytesIO(image_bytes))
        results = self.model.predict(image, save=False, verbose=False, conf=0.01)
        
        dict_result = []
        
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidence = result.boxes.conf
            
            for box, cls, conf in zip(boxes, classes, confidence):
                dict_result.append({
                    'class': int(cls),
                    'class_name': self.model.names[int(cls)],
                    'confidence': float(conf),
                    'box': [float(coord) for coord in box]
                })
                
        return dict_result
        