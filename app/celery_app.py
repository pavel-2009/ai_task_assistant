"""
Конфигурация и инициализация Celery для асинхронного выполнения задач в фоновом режиме.
"""

from celery import Celery
import os

from app.ml.yolo_service import YoloService


CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")


celery_app = Celery(
    "ai_task_assistant",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.ml.tasks"]
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


yolo_service = YoloService()