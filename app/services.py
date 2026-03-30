"""
Глобальный реестр сервисов. Инициализируется в lifespan приложения.
Доступен как FastAPI, так и Celery воркерам.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import redis.asyncio as redis
from redis.exceptions import RedisError
from sqlalchemy import select

from app.core import config
from app.db import async_session
from app.db_models import Task

from app.ml.cv.classification.inference_service import InferenceService
from app.ml.cv.detection.yolo_service import YoloService
from app.ml.cv.detection.yolo_onnx_service import YoloONNXService
from app.ml.cv.segmentation.segmentation_service import SegmentationService
from app.ml.cv.embedding.image_embedding_service import ImageEmbeddingService
from app.ml.nlp.rag_service import RAGService
from app.ml.nlp.vector_db import VectorDB
from app.ml.nlp.ner_service import NerService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.llm_service import LLMService
from app.ml.recsys.content_based import ContentBasedRecommender
from app.ml.recsys.collaborative_filtering import CollaborativeFilteringRecommender
from app.ml.recsys.vector_db.recsys_vector_db import RecSysVectorDB
from app.ml.monitoring.drift_detector import DriftDetector

logger = logging.getLogger(__name__)


_services: dict = {}
_initialized = False
_init_lock = asyncio.Lock()


def default_inference_checkpoint_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "checkpoints" / "model.pth")


def default_inference_idx_to_class() -> dict[int, str]:
    return {0: "cat", 1: "dog", 2: "house"}


async def init_services(
    use_onnx: bool = False,
    redis_client: redis.Redis | None = None,
    inference_checkpoint_path: str = None,
    inference_idx_to_class: dict = None,
) -> None:
    """Инициализирует все сервисы один раз при старте приложения."""
    global _initialized

    if _initialized:
        return

    # Redis
    if redis_client is None:
        try:
            redis_client = redis.from_url(config.REDIS_URL)
            await redis_client.ping()
        except RedisError:
            redis_client = None
            logger.warning("Не удалось подключиться к Redis. Некоторые функции будут недоступны.")
        except Exception as e:
            redis_client = None
            logger.error(f"Ошибка при инициализации Redis: {e}")
        
    _services["redis"] = redis_client
    
    # CV сервисы
    _services["inference"] = InferenceService(
        checkpoints_path=inference_checkpoint_path,
        idx_to_class=inference_idx_to_class or {}
    )
    
    _services["image_embedding"] = ImageEmbeddingService()
    
    if use_onnx:
        _services["yolo"] = YoloONNXService()
    else:
        _services["yolo"] = YoloService()
    
    _services["segmentation"] = SegmentationService()
    
    # NLP сервисы
    _services["embedding"] = EmbeddingService()
    _services["ner"] = NerService()
    _services["llm"] = LLMService()
    await _services["llm"].warmup()
    
    # Vector DB и семантический поиск
    _services["vector_db"] = VectorDB(
        dim=_services["embedding"].dimension,
        redis_client=redis_client
    )
    
    # Рекомендательная система
    _services["recsys_vector_db"] = RecSysVectorDB(dim=896, redis_client=redis_client)  
    
    _services["content_based_recommender"] = ContentBasedRecommender(
        image_embedding_service=_services["image_embedding"],
        text_embedding_service=_services["embedding"],
        image_vector_db=_services["recsys_vector_db"]
    )
    
    _services["collaborative_filtering_recommender"] = CollaborativeFilteringRecommender(
        redis_client=redis_client
    )


    if redis_client:
        await _services["vector_db"].load_from_redis()
    
    _services["semantic_search"] = SemanticSearchService(
        embedding_service=_services["embedding"],
        vector_db=_services["vector_db"],
        redis_client=redis_client,
    )
    
    # RAG сервис
    _services["rag"] = RAGService(
        llm_service=_services["llm"],
        semantic_search_service=_services["semantic_search"],
        redis=redis_client,
    )
    
    # Детектор дрейфа
    _services["drift_detector"] = DriftDetector()
    
    # Загружаем референсные эмбеддинги всех существующих аватаров
    async with async_session() as session:
        tasks = await session.execute(select(Task).where(Task.avatar_file.isnot(None)))
        reference_embeddings = []
        for task in tasks.scalars():
            if os.path.exists(task.avatar_file):
                with open(task.avatar_file, "rb") as f:
                    emb = _services["image_embedding"].get_embedding(f.read())
                    reference_embeddings.append(emb)
        
        if reference_embeddings:
            for emb in reference_embeddings:
                _services["drift_detector"].add_embedding(emb)

    _initialized = True


async def ensure_services_initialized(
    use_onnx: bool = False,
    redis_client: redis.Redis | None = None,
    inference_checkpoint_path: str | None = None,
    inference_idx_to_class: dict | None = None,
) -> None:
    """Потокобезопасная инициализация сервисов для FastAPI и Celery."""
    if _initialized:
        return

    async with _init_lock:
        if _initialized:
            return

        await init_services(
            use_onnx=use_onnx,
            redis_client=redis_client,
            inference_checkpoint_path=inference_checkpoint_path or default_inference_checkpoint_path(),
            inference_idx_to_class=inference_idx_to_class or default_inference_idx_to_class(),
        )


def get_service(name: str) -> object:
    """Получить инициализированный сервис."""
    
    if name not in _services:
        raise RuntimeError(
            f"Сервис '{name}' не инициализирован. "
            f"Доступные сервисы: {list(_services.keys())}"
        )
    return _services[name]


def get_redis() -> redis.Redis:
    return get_service("redis")

def get_inference() -> InferenceService:
    return get_service("inference")

def get_yolo() -> YoloService | YoloONNXService:
    return get_service("yolo")

def get_segmentation() -> SegmentationService:
    return get_service("segmentation")

def get_image_embedding() -> ImageEmbeddingService:
    return get_service("image_embedding")

def get_embedding() -> EmbeddingService:
    return get_service("embedding")

def get_ner() -> NerService:
    return get_service("ner")

def get_llm() -> LLMService:
    return get_service("llm")

def get_vector_db() -> VectorDB:
    return get_service("vector_db")

def get_semantic_search() -> SemanticSearchService:
    return get_service("semantic_search")

def get_rag() -> RAGService:
    return get_service("rag")

def get_content_based_recommender() -> ContentBasedRecommender:
    return get_service("content_based_recommender")

def get_recsys_vector_db() -> RecSysVectorDB:
    return get_service("recsys_vector_db")

def get_collaborative_filtering_recommender() -> CollaborativeFilteringRecommender:
    return get_service("collaborative_filtering_recommender")

def get_drift_detector() -> DriftDetector:
    return get_service("drift_detector")
