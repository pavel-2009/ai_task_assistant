"""
Конфигурация и инициализация Celery для асинхронного выполнения задач в фоновом режиме.
"""

from celery import Celery
from celery.schedules import crontab
import os
from dotenv import load_dotenv


load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")


celery_app = Celery(
    "ai_task_assistant",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "app.ml.cv.tasks",
        "app.ml.nlp.tasks",
        "app.ml.recsys.tasks"
    ]
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
)

celery_app.conf.beat_schedule = {
    "reindex-tasks-every-day": {
        "task": "train_collaborative_filtering_model",
        "schedule": crontab(hour=0, minute=0),  # Каждый день в полночь
    },
}
