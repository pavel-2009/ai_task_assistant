"""
Постпроцессинг результатов инференса и NMS для YOLO ONNX
"""

import numpy as np
import time
from .config import COCO_CLASSES, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD

class OutputPostprocessor:
    """Постпроцессинг результатов инференса с NMS"""
    
    def __init__(self):
        self.postprocess_times = []
    
    def postprocess(self, output: list[np.ndarray], scale: float, pad_x: int, pad_y: int, 
                   orig_shape: tuple[int, int], conf_threshold: float = DEFAULT_CONF_THRESHOLD, 
                   iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> list[dict]:
        """Ускоренный постпроцессинг с векторизацией"""
        start_time = time.time()
        
        conf_threshold = float(conf_threshold)
        iou_threshold = float(iou_threshold)

        # Изменяем порядок для векторизации (84, 8400) вместо (8400, 84)
        preds = output[0].squeeze(0)  # (84, 8400)
        
        # Векторизованное получение максимальных скорингов
        class_scores = preds[4:, :]  # (80, 8400)
        max_scores = np.max(class_scores, axis=0)  # (8400,)
        class_ids = np.argmax(class_scores, axis=0)  # (8400,)
        
        # Быстрая фильтрация по порогу
        mask = max_scores > conf_threshold
        if not np.any(mask):
            return []

        # Применяем маску ко всем данным сразу
        boxes = preds[:4, mask].T  # (n, 4)
        confidences = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Конвертация координат (векторизовано)
        x_c, y_c, w, h = boxes.T
        
        # Обратное масштабирование за один проход
        x1 = (x_c - w/2 - pad_x) / scale
        y1 = (y_c - h/2 - pad_y) / scale
        x2 = (x_c + w/2 - pad_x) / scale
        y2 = (y_c + h/2 - pad_y) / scale
        
        # Клиппинг к границам изображения
        x1 = np.clip(x1, 0, orig_shape[1])
        y1 = np.clip(y1, 0, orig_shape[0])
        x2 = np.clip(x2, 0, orig_shape[1])
        y2 = np.clip(y2, 0, orig_shape[0])
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Быстрый NMS
        indices = self._fast_nms(boxes, confidences, iou_threshold)

        results = []
        if len(indices) > 0:
            for i in indices:
                results.append({
                    "class_name": COCO_CLASSES[int(class_ids[i])],
                    "confidence": float(confidences[i]),
                    "box": boxes[i].tolist()
                })
        
        postprocess_time = time.time() - start_time
        self.postprocess_times.append(postprocess_time)
        if len(self.postprocess_times) > 100:
            self.postprocess_times.pop(0)
        
        return results
    
    def _fast_nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """Ускоренный NMS без OpenCV"""
        if len(boxes) == 0:
            return []
        
        # Сортируем по уверенности
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Вычисляем IoU для всех оставшихся
            ious = self._compute_iou(boxes[i], boxes[order[1:]])
            
            # Оставляем только те, у которых IoU < порога
            mask = ious <= iou_threshold
            order = order[1:][mask]
        
        return keep

    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Векторизованное вычисление IoU"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = area1 + area2 - intersection
        return intersection / (union + 1e-6)
