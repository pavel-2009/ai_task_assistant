"""
Конфигурация и инициализация Celery для асинхронного выполнения задач в фоновом режиме.
"""

from celery import Celery
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

from app.ml.inference_service import InferenceService
from app.ml.yolo_service import YoloService
from app.ml.segmentation_service import SegmentationService


# Кэш для моделей и executor для асинхронной загрузки
_models_cache = {"inference": None, "yolo": None, "segmentation": None}
_executor = ThreadPoolExecutor(max_workers=3)



CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

INFERENCE_CHECKPOINT_PATH = Path(__file__).parent.parent /  "checkpoints" / "model.pth"
INFERENCE_IDX_TO_CLASS = {0: "cat", 1: "dog", 2: "house"}


def get_inference_service() -> InferenceService:
    if _models_cache["inference"] is None:
        _models_cache["inference"] = InferenceService(INFERENCE_CHECKPOINT_PATH, INFERENCE_IDX_TO_CLASS)
    return _models_cache["inference"]


def get_yolo_service() -> YoloService:
    if _models_cache["yolo"] is None:
        _models_cache["yolo"] = YoloService()
    return _models_cache["yolo"]


def get_segmentation_service() -> SegmentationService:
    if _models_cache["segmentation"] is None:
        _models_cache["segmentation"] = SegmentationService()
    return _models_cache["segmentation"]


async def preload_models():
    """Загрузка всех моделей при старте приложения в фоновом режиме"""
    loop = asyncio.get_event_loop()
    
    tasks = [
        loop.run_in_executor(_executor, get_inference_service),
        loop.run_in_executor(_executor, get_yolo_service),
        loop.run_in_executor(_executor, get_segmentation_service)
    ]
    
    await asyncio.gather(*tasks)
    print("Все модели загружены.")


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
