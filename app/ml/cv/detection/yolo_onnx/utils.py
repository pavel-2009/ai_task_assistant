"""
Утилиты для YOLO ONNX сервиса (кэширование, метрики, визуализация)
"""

import hashlib
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from .config import CACHE_SIZE, VISUALIZATION_DIR

class FrameCache:
    """Кэширование результатов для похожих кадров"""
    
    def __init__(self, cache_size: int = CACHE_SIZE):
        self.cache: Dict[str, List[dict]] = {}
        self.cache_size = cache_size
    
    def get_frame_hash(self, image_bytes: bytes) -> str:
        """Быстрый хеш для похожих кадров"""
        if len(image_bytes) > 10000:
            return hashlib.md5(image_bytes[:10000]).hexdigest()
        return hashlib.md5(image_bytes).hexdigest()
    
    def get(self, frame_hash: str) -> List[dict] | None:
        """Получить результат из кэша"""
        return self.cache.get(frame_hash)
    
    def set(self, frame_hash: str, results: List[dict]) -> None:
        """Сохранить результат в кэш"""
        self.cache[frame_hash] = results
        # Удаляем старый элемент если кэш переполнен
        if len(self.cache) > self.cache_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
    
    def clear(self):
        """Очистить кэш"""
        self.cache.clear()


class MetricsCollector:
    """Сбор и анализ метрик производительности"""
    
    def __init__(self, window_size: int = 100):
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        self.window_size = window_size
    
    def add_inference_time(self, time: float):
        """Добавить время инференса"""
        self.inference_times.append(time)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)
    
    def add_preprocess_time(self, time: float):
        """Добавить время препроцессинга"""
        self.preprocess_times.append(time)
        if len(self.preprocess_times) > self.window_size:
            self.preprocess_times.pop(0)
    
    def add_postprocess_time(self, time: float):
        """Добавить время постпроцессинга"""
        self.postprocess_times.append(time)
        if len(self.postprocess_times) > self.window_size:
            self.postprocess_times.pop(0)
    
    def get_average_times(self) -> Tuple[float, float, float]:
        """Получить средние времена"""
        pre = np.mean(self.preprocess_times) if self.preprocess_times else 0
        inf = np.mean(self.inference_times) if self.inference_times else 0
        post = np.mean(self.postprocess_times) if self.postprocess_times else 0
        return pre, inf, post
    
    def log_performance(self):
        """Логирование текущей производительности"""
        pre, inf, post = self.get_average_times()
        total = pre + inf + post
        print(f"Performance - Pre: {pre:.3f}s, Inf: {inf:.3f}s, Post: {post:.3f}s, Total: {total:.3f}s")


class VisualizationHelper:
    """Вспомогательные функции для визуализации результатов"""
    
    def __init__(self, output_dir: Path = VISUALIZATION_DIR):
        self.output_dir = output_dir
    
    def visualize(self, image_bytes: bytes, detections: List[dict], task_id: int) -> Path:
        """Визуализация результатов детекции на изображение"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cls_name = det["class_name"]
            conf = det["confidence"]
            label = f"{cls_name} {conf:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        output_path = self.output_dir / f"result_{task_id}_onnx.jpg"
        cv2.imwrite(str(output_path), img)
        
        return output_path
