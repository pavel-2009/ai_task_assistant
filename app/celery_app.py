"""
Конфигурация и инициализация Celery для асинхронного выполнения задач в фоновом режиме.
"""

from celery import Celery
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from dotenv import load_dotenv

from app.ml.cv.classification.inference_service import InferenceService
from app.ml.cv.detection.yolo_service import YoloService
from app.ml.cv.segmentation.segmentation_service import SegmentationService
from app.ml.cv.detection.yolo_onnx_service import YoloONNXService
from app.ml.nlp.rag_service import RAGService
from app.ml.nlp.vector_db import VectorDB
from app.ml.nlp.ner_service import NerService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.embedding_service import EmbeddingService


load_dotenv()


# Кэш для моделей и executor для асинхронной загрузки
_models_cache = {
    "inference": None,
    "yolo": None,
    "segmentation": None,
    "rag": None,
    "vectordb": None,
    "semantic_search": None,
    "ner": None,
    "embedding": None
}
_executor = ThreadPoolExecutor(max_workers=3)



CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

INFERENCE_CHECKPOINT_PATH = Path(__file__).parent.parent /  "checkpoints" / "model.pth"
INFERENCE_IDX_TO_CLASS = {0: "cat", 1: "dog", 2: "house"}

USE_ONNX = os.getenv("USE_ONNX", "False").lower() in ("true", "1", "t")


def get_inference_service() -> InferenceService:
    if _models_cache["inference"] is None:
        _models_cache["inference"] = InferenceService(INFERENCE_CHECKPOINT_PATH, INFERENCE_IDX_TO_CLASS)
    return _models_cache["inference"]


def get_yolo_service() -> YoloService:
    if _models_cache["yolo"] is None:
        if USE_ONNX:
            _models_cache["yolo"] = YoloONNXService()
        else:
            _models_cache["yolo"] = YoloService()
    return _models_cache["yolo"]


def get_segmentation_service() -> SegmentationService:
    if _models_cache["segmentation"] is None:
        _models_cache["segmentation"] = SegmentationService()
    return _models_cache["segmentation"]


def get_rag_service():
    if _models_cache["rag"] is None:
        _models_cache["rag"] = RAGService()
    return _models_cache["rag"]


def get_vector_db():
    if _models_cache["vectordb"] is None:
        _models_cache["vectordb"] = VectorDB()
    return _models_cache["vectordb"]


def get_semantic_search_service():
    if _models_cache["semantic_search"] is None:
        _models_cache["semantic_search"] = SemanticSearchService()
    return _models_cache["semantic_search"]


def get_ner_service():
    if _models_cache["ner"] is None:
        _models_cache["ner"] = NerService()
    return _models_cache["ner"]


def get_embedding_service():
    if _models_cache["embedding"] is None:
        _models_cache["embedding"] = EmbeddingService()
    return _models_cache["embedding"]


async def preload_models():
    """Загрузка всех моделей при старте приложения в фоновом режиме"""
    loop = asyncio.get_event_loop()
    
    tasks = [
        loop.run_in_executor(_executor, get_inference_service),
        loop.run_in_executor(_executor, get_yolo_service),
        loop.run_in_executor(_executor, get_segmentation_service),
        loop.run_in_executor(_executor, get_rag_service),
        loop.run_in_executor(_executor, get_vector_db),
        loop.run_in_executor(_executor, get_semantic_search_service),
        loop.run_in_executor(_executor, get_ner_service),
        loop.run_in_executor(_executor, get_embedding_service)
    ]
    
    await asyncio.gather(*tasks)
    print("Все модели загружены.")


celery_app = Celery(
    "ai_task_assistant",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "app.ml.cv.tasks",
        "app.ml.nlp.tasks"
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
