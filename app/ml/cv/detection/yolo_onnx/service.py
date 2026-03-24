"""
Основной сервис YOLO ONNX инференса
"""

import onnxruntime as ort
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from .config import (
    ONNX_WEIGHTS_PATH, get_session_options, 
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, THREADPOOL_WORKERS
)
from .preprocessing import ImagePreprocessor
from .postprocessing import OutputPostprocessor
from .utils import FrameCache, MetricsCollector, VisualizationHelper


class YoloONNXService:
    """ONNX-сервис для инференса YOLOv8 с возвратом имени класса"""

    def __init__(self, model_path: str | Path = ONNX_WEIGHTS_PATH):

        self.model_path = Path(model_path)
        self.executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)
        
        # Компоненты обработки
        self.preprocessor = ImagePreprocessor(model_size=640)
        self.postprocessor = OutputPostprocessor()
        self.cache = FrameCache()
        self.metrics = MetricsCollector()
        self.visualizer = VisualizationHelper()
        
        # Инициализация ONNX модели
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=get_session_options(),
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.model_size = self.input_shape[2]  # 640 для yolov8n
    
    def predict(
        self,
        image_bytes: bytes,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD
    ) -> List[dict]:
        """Получение предсказаний модели для изображения"""
        start_time = time.time()
        
        # Проверяем кэш
        frame_hash = self.cache.get_frame_hash(image_bytes)
        cached = self.cache.get(frame_hash)
        if cached is not None:
            return cached
        
        # Инференс
        input_tensor, scale, pad_x, pad_y, orig_shape = self.preprocessor.preprocess(image_bytes)
        
        inference_start = time.time()
        output = self.session.run(None, {self.input_name: input_tensor})
        inference_time = time.time() - inference_start
        self.metrics.add_inference_time(inference_time)
        
        results = self.postprocessor.postprocess(
            output,
            scale,
            pad_x,
            pad_y,
            orig_shape,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Обновляем кэш
        self.cache.set(frame_hash, results)
        
        # Периодическое логирование
        if len(self.metrics.inference_times) % 30 == 0:
            self.metrics.log_performance()
        
        return results
    
    async def predict_async(self, image_bytes: bytes, conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                            iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> List[dict]:
        """Асинхронная версия predict"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.predict, 
            image_bytes,
            conf_threshold,
            iou_threshold
        )

    async def predict_batch_async(self, images_bytes: List[bytes],
                                  conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                                  iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> List[List[dict]]:
        """Батч-обработка нескольких кадров"""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.predict, img_bytes, conf_threshold, iou_threshold)
            for img_bytes in images_bytes
        ]
        return await asyncio.gather(*tasks)

    def predict_without_postprocess(self, image_bytes: bytes) -> List:
        """Предсказание без постпроцессинга (сырые данные)"""
        input_tensor, _, _, _, _ = self.preprocessor.preprocess(image_bytes)
        output = self.session.run(None, {self.input_name: input_tensor})
        return output

    def predict_and_visualize(self, image_bytes: bytes, task_id: int,
                              conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                              iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> tuple[List[dict], Path]:
        """Предсказание с визуализацией результатов"""
        results = self.predict(image_bytes, conf_threshold, iou_threshold)
        output_path = self.visualizer.visualize(image_bytes, results, task_id)
        return results, output_path

    def clear_cache(self):
        """Очистить кэш результатов"""
        self.cache.clear()

    def __del__(self):
        """Освобождение ресурсов"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
