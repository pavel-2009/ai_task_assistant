"""
ONNX-сервис для YOLOv8 с возвратом имени класса
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import os
from dotenv import load_dotenv
from functools import lru_cache
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

load_dotenv()

ONNX_WEIGHTS_PATH = Path(__file__).parent / "yolov8n.onnx"
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

# =================== ОПТИМИЗАЦИЯ 1: Настройки сессии для максимальной производительности
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = os.cpu_count()
sess_options.inter_op_num_threads = 1
sess_options.enable_cpu_mem_arena = False  # Отключаем для скорости
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
# ===================

class YoloONNXService:
    """ONNX-сервис для инференса YOLOv8 с возвратом имени класса"""

    def __init__(self, model_path: str = ONNX_WEIGHTS_PATH):
        # =================== ОПТИМИЗАЦИЯ 2: Кэши и пулы потоков
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.cache = {}
        self.cache_size = 5
        self.use_fp16 = self._check_fp16_support()
        self.metrics = {'inference_times': [], 'preprocess_times': [], 'postprocess_times': []}
        # ===================
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.model_size = self.input_shape[2]  # 640 для yolov8n
        
    # =================== ОПТИМИЗАЦИЯ 3: Проверка поддержки FP16
    def _check_fp16_support(self):
        """Проверяет, поддерживает ли CPU FP16 инструкции"""
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            flags = cpu_info.get('flags', [])
            return 'f16c' in flags or 'avx' in flags
        except:
            return False
    # ===================

    # =================== ОПТИМИЗАЦИЯ 4: Кэширование letterbox для одинаковых размеров
    @lru_cache(maxsize=32)
    def _get_letterbox_params(self, h: int, w: int):
        """Кэширование параметров letterbox для одинаковых размеров кадра"""
        scale = min(self.model_size / w, self.model_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_left = (self.model_size - new_w) // 2
        pad_top = (self.model_size - new_h) // 2
        return scale, new_w, new_h, pad_left, pad_top
    # ===================

    # =================== ОПТИМИЗАЦИЯ 5: Ускоренная предобработка
    def preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, float, int, int, tuple[int, int]]:
        """Преобразование изображения для модели (оптимизированная версия)"""
        start_time = time.time()
        
        # Быстрое декодирование JPEG
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        orig_shape = (h, w)
        
        # Используем кэшированные параметры
        scale, new_w, new_h, pad_left, pad_top = self._get_letterbox_params(h, w)
        
        # Оптимизированный ресайз с интерполяцией
        if (new_w, new_h) != (w, h):
            interpolation = cv2.INTER_LINEAR if scale > 0.5 else cv2.INTER_AREA
            resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        else:
            resized = img
        
        # Быстрое создание канвы с использованием numpy broadcasting
        canvas = np.full((self.model_size, self.model_size, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        
        # Нормализация и transpose за один проход
        canvas = canvas / 255.0
        
        # HWC -> CHW с добавлением batch dimension
        input_tensor = np.expand_dims(canvas.transpose(2, 0, 1), axis=0)
        
        preprocess_time = time.time() - start_time
        self.metrics['preprocess_times'].append(preprocess_time)
        if len(self.metrics['preprocess_times']) > 100:
            self.metrics['preprocess_times'].pop(0)
        
        return input_tensor, scale, pad_left, pad_top, orig_shape
    # ===================

    # =================== ОПТИМИЗАЦИЯ 6: Векторизованный постпроцессинг
    def postprocess(self, output: list[np.ndarray], scale: float, pad_x: int, pad_y: int, 
                   orig_shape: tuple[int, int], conf_threshold: float = 0.45, 
                   iou_threshold: float = 0.55):
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
        self.metrics['postprocess_times'].append(postprocess_time)
        if len(self.metrics['postprocess_times']) > 100:
            self.metrics['postprocess_times'].pop(0)
        
        return results
    # ===================

    # =================== ОПТИМИЗАЦИЯ 7: Векторизованный NMS
    def _fast_nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
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
    # ===================

    # =================== ОПТИМИЗАЦИЯ 8: Кэширование результатов для похожих кадров
    def _get_frame_hash(self, image_bytes: bytes) -> str:
        """Быстрый хеш для похожих кадров"""
        if len(image_bytes) > 10000:
            return hashlib.md5(image_bytes[:10000]).hexdigest()
        return hashlib.md5(image_bytes).hexdigest()

    def predict(self, image_bytes: bytes) -> list[dict]:
        """Предсказание с постпроцессингом и кэшированием"""
        start_time = time.time()
        
        # Проверяем кэш
        frame_hash = self._get_frame_hash(image_bytes)
        if frame_hash in self.cache:
            return self.cache[frame_hash]
        
        # Инференс
        input_tensor, scale, pad_x, pad_y, orig_shape = self.preprocess(image_bytes)
        
        inference_start = time.time()
        output = self.session.run(None, {self.input_name: input_tensor})
        inference_time = time.time() - inference_start
        
        self.metrics['inference_times'].append(inference_time)
        if len(self.metrics['inference_times']) > 100:
            self.metrics['inference_times'].pop(0)
        
        results = self.postprocess(
            output, scale, pad_x, pad_y, orig_shape,
            conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", 0.35)),
            iou_threshold=float(os.getenv("YOLO_IOU_THRESHOLD", 0.55))
        )
        
        # Обновляем кэш
        self.cache[frame_hash] = results
        if len(self.cache) > self.cache_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        
        total_time = time.time() - start_time
        if len(self.metrics['inference_times']) % 30 == 0:
            print(f"Performance - Pre: {np.mean(self.metrics['preprocess_times']):.3f}s, "
                  f"Inf: {np.mean(self.metrics['inference_times']):.3f}s, "
                  f"Post: {np.mean(self.metrics['postprocess_times']):.3f}s, "
                  f"Total: {total_time:.3f}s")
        
        return results
    # ===================

    # =================== ОПТИМИЗАЦИЯ 9: Асинхронный predict для вебсокетов
    async def predict_async(self, image_bytes: bytes) -> list[dict]:
        """Асинхронная версия predict"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.predict, image_bytes)

    async def predict_batch_async(self, images_bytes: list[bytes]) -> list[list[dict]]:
        """Батч-обработка нескольких кадров"""
        loop = asyncio.get_event_loop()
        tasks = []
        for img_bytes in images_bytes:
            tasks.append(loop.run_in_executor(self.executor, self.predict, img_bytes))
        return await asyncio.gather(*tasks)
    # ===================

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

    # =================== ОПТИМИЗАЦИЯ 10: Очистка ресурсов
    def __del__(self):
        """Освобождение ресурсов"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    # ===================