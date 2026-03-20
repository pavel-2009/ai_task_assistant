"""
Препроцессинг изображений для YOLO ONNX
"""

import numpy as np
import cv2
import time
from functools import lru_cache


class ImagePreprocessor:
    """Препроцессинг изображений с оптимизацией"""
    
    def __init__(self, model_size: int = 640):
        self.model_size = model_size
        self.preprocess_times = [] # Для сбора метрик времени препроцессинга
    
    @lru_cache(maxsize=32)
    def _get_letterbox_params(self, h: int, w: int):
        """Кэширование параметров letterbox для одинаковых размеров кадра"""
        scale = min(self.model_size / w, self.model_size / h) 
        new_w, new_h = int(w * scale), int(h * scale)
        
        pad_left = (self.model_size - new_w) // 2
        pad_top = (self.model_size - new_h) // 2
        
        return scale, new_w, new_h, pad_left, pad_top
    
    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, float, int, int, tuple[int, int]]:
        """Препроцессинг изображения для модели YOLO ONNX"""
        start_time = time.time()
        
        # Быстрое декодирование JPEG
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        orig_shape = (h, w)
        
        # Используем кэшированные параметры
        scale, new_w, new_h, pad_left, pad_top = self._get_letterbox_params(h, w)
        
        # Изменяем размер с оптимизацией интерполяции
        if (new_w, new_h) != (w, h):
            interpolation = cv2.INTER_LINEAR if scale > 0.5 else cv2.INTER_AREA # Выбираем интерполяцию (то есть метод) в зависимости от масштаба
            
            # На выходе - оптимизированное изменение размера с минимальными потерями качества
            resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
            
        else:
            resized = img
        
        # Быстрое создание канвы с использованием numpy broadcasting
        canvas = np.full((self.model_size, self.model_size, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        
        # Нормализация и transpose за один проход
        canvas = canvas.astype(np.float32) / 255.0
        
        # HWC -> CHW с добавлением batch dimension
        input_tensor = np.expand_dims(canvas.transpose(2, 0, 1), axis=0) # expand_dims добавляет размерность для батча
        
        preprocess_time = time.time() - start_time
        self.preprocess_times.append(preprocess_time)
        if len(self.preprocess_times) > 100:
            self.preprocess_times.pop(0)
        
        return input_tensor, scale, pad_left, pad_top, orig_shape
