"""
Асинхронные задачи для обработки изображений с помощью модели YOLO.
"""

from app.celery_app import celery_app, yolo_service



@celery_app.task(name="detect_and_visualize_task")
def detect_and_visualize_task(task_id: int, image_path: str) -> dict:
    """Фоновая задача для обнаружения объектов на изображении и сохранения визуализации."""
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    detections, visualized_image_path = yolo_service.predict_and_visualize(image_bytes, task_id)
    
    return {"detections": detections, "visualized_image": visualized_image_path}