"""
Асинхронные задачи для обработки изображений с помощью модели YOLO.
"""

from app.celery_app import celery_app, yolo_service, segmentation_service

from pathlib import Path
import logging

logger = logging.getLogger(__name__)



@celery_app.task(name="detect_and_visualize_task")
def detect_and_visualize_task(task_id: int, image_path: str) -> dict:
    """Фоновая задача для обнаружения объектов на изображении и сохранения визуализации."""
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    detections, visualized_image_path = yolo_service.predict_and_visualize(image_bytes, task_id)
    
    return {"detections": detections, "visualized_image": str(visualized_image_path)}


@celery_app.task(name="segment_image_task")
def segment_image_task(task_id: int, image_path: str) -> bytes:
    """Фоновая задача для сегментации изображения."""
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        segmented_image_bytes = segmentation_service.segment_image(image_bytes)
        
        path_to_save = Path("/app/avatars/segments") / f"{task_id}_segmentation.png"
        
        logger.info(f"Saving segmentation result to: {path_to_save}")
        
        if not path_to_save.parent.exists():
            logger.info(f"Creating directory: {path_to_save.parent}")
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_to_save, "wb") as f:
            f.write(segmented_image_bytes)
        
        logger.info(f"✅ Segmentation saved successfully: {path_to_save}")
        
        return segmented_image_bytes
    
    except Exception as e:
        logger.error(f"❌ Error in segment_image_task: {str(e)}", exc_info=True)
        raise