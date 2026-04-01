"""
Асинхронные задачи для обработки изображений с помощью модели YOLO.
"""

import logging

from app.celery_app import celery_app
from app.celery_metrics import track_celery_task
from app.services import (
    get_inference,
    get_segmentation,
    get_yolo,
    get_image_embedding,
    get_drift_detector,
    get_redis
)

from pathlib import Path
from datetime import datetime

import redis


logger = logging.getLogger(__name__)


@celery_app.task(name="predict_avatar_class")
@track_celery_task("predict_avatar_class")
def predict_avatar_class(task_id: int, image_path: str) -> dict:
    """Фоновая задача для предсказания класса аватарки задачи."""
    
    try:
        inference_service = get_inference()
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        predicted_class = inference_service.predict(image_bytes)
        
        return predicted_class
    except RuntimeError as e:
        logger.warning(f"Could not predict avatar class (service may not be initialized): {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error during avatar prediction: {e}", exc_info=True)
        return {"error": str(e)}


@celery_app.task(name="detect_and_visualize_task")
@track_celery_task("detect_and_visualize_task")
def detect_and_visualize_task(task_id: int, image_path: str) -> dict:
    """Фоновая задача для обнаружения объектов на изображении и сохранения визуализации."""
    
    try:
        yolo_service = get_yolo()
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        detections, visualized_image_path = yolo_service.predict_and_visualize(image_bytes, task_id)
        
        return {"detections": detections, "visualized_image": str(visualized_image_path)}
    except RuntimeError as e:
        logger.warning(f"Could not detect and visualize task (service may not be initialized): {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error during object detection: {e}", exc_info=True)
        return {"error": str(e)}


@celery_app.task(name="segment_image_task")
def segment_image_task(task_id: int, image_path: str) -> bytes:
    """Фоновая задача для сегментации изображения."""
    
    segmentation_service = get_segmentation()
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        segmented_image_bytes = segmentation_service.segment_image(image_bytes)
        
        path_to_save = Path("/app/avatars/segments") / f"{task_id}_segmentation.png"
        
        if not path_to_save.parent.exists():
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_to_save, "wb") as f:
            f.write(segmented_image_bytes)
        
        return segmented_image_bytes
    
    except Exception as e:
        raise
    
    
@celery_app.task(name="check_avatar_drift")
def check_avatar_drift(image_path: str):
    """Фоновая проверка дрейфа для загруженного аватара"""
    
    embedding_service = get_image_embedding()
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        
    embedding = embedding_service.get_embedding(image_bytes)
    
    detector = get_drift_detector()
    
    detector.add_embedding(embedding)
    
    drift_result = detector.calculate_drift([embedding])
    
    if drift_result["drift_detected"]:
        redis_client: redis.Redis = get_redis()
        
        redis_client.set("drift_detected", f"{datetime.now()}: {drift_result}")
