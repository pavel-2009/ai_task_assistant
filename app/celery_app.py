"""
Конфигурация и инициализация Celery для асинхронного выполнения задач в фоновом режиме.
"""

import asyncio
import logging

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init

from app.core import config
from app.services import ensure_services_initialized

logger = logging.getLogger(__name__)


celery_app = Celery(
    "ai_task_assistant",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
    include=[
        "app.ml.cv.tasks",
        "app.ml.nlp.tasks",
        "app.ml.recsys.tasks",
    ],
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
        "schedule": crontab(hour=0, minute=0),
    },
}


@worker_process_init.connect
def init_services_for_worker(**kwargs):
    """Единая инициализация сервисов для каждого процесса Celery worker."""
    try:
        asyncio.run(ensure_services_initialized(use_onnx=config.USE_ONNX))
        logger.info("Celery services initialized")
    except Exception as exc:
        logger.error("Celery services initialization failed: %s", exc, exc_info=True)
