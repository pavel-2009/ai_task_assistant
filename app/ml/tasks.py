"""
Асинхронные задачи для обработки изображений с помощью модели YOLO.
"""

from app.celery_app import celery_app



@celery_app.task(name="detect_and_visualize_task")
def detect_and_visualize_task(task_id: int, image_path: str) -> dict:
    """Фоновая задача для обнаружения объектов на изображении и сохранения визуализации."""
    
    pass
    