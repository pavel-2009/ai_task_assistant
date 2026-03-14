"""
Сервис для работы с моделью YOLOv8.
"""

from ultralytics import YOLO
from PIL import Image
import io

from pathlib import Path


MODEL_PATH = 'yolov8n.pt'
VISUALIZATION_DIR = Path(__file__).parent.parent.parent / 'avatars' / 'visualizations'
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR = Path(__file__).parent.parent.parent / "checkpoints" / "onnx"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class YoloService:
    """Сервис для инференса модели YOLOv8."""
    
    def __init__(self, model_path: Path | str = MODEL_PATH):
        """Инициализация модели."""
        self.model = YOLO(model_path)
        self.model.to('cpu')
        
        
    def predict_and_visualize(self, image_bytes: bytes, task_id: int) -> list[dict]:
        """Получения предсказаний модели для изображения"""
        
        image = Image.open(io.BytesIO(image_bytes))
        results = self.model.predict(image, save=False, verbose=False, conf=0.45)
        
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
                
        im = results[0].plot()
                
        Image.fromarray(im).save(VISUALIZATION_DIR / f'result_{task_id}.jpg')
                
        return dict_result, VISUALIZATION_DIR / f'result_{task_id}.jpg'
    
    
    def export_onnx(self):
        """Экспорт модели в ONNX"""
        
        onnx_path = self.model.export(
            format="onnx",
            imgsz=640,
            dynamic=False,
            half=False,
            verbose=False,
            project=str(EXPORT_DIR),
            name="yolov8n_onnx"
        )
        
        return onnx_path
    

if __name__ == '__main__':
    service = YoloService('yolov8n.pt')
    
    onnx_path = service.export_onnx()
    
    print(f"Успешно экспортированно в {onnx_path}")