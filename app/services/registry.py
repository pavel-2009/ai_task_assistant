"""Глобальный реестр ML-сервисов, инициализируемый при запуске приложения."""

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
from app.ml.cv.embedding.image_embedding_service import ImageEmbeddingService
from app.ml.cv.segmentation.segmentation_service import SegmentationService
from app.ml.monitoring.drift_detector import DriftDetector
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.llm_service import LLMService
from app.ml.nlp.ner_service import NerService
from app.ml.nlp.rag_service import RAGService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.vector_db import VectorDB
from app.ml.recsys.collaborative_filtering import CollaborativeFilteringRecommender
from app.ml.recsys.content_based import ContentBasedRecommender
from app.ml.recsys.vector_db.recsys_vector_db import RecSysVectorDB

logger = logging.getLogger(__name__)

_services: dict[str, object] = {}
_initialized = False
_init_lock = asyncio.Lock()


def default_inference_checkpoint_path() -> str:
    return str(Path(__file__).resolve().parent.parent.parent / "checkpoints" / "model.pth")


async def init_services(
    use_onnx: bool = False,
    redis_client: redis.Redis | None = None,
    inference_checkpoint_path: str | None = None,
    inference_idx_to_class: dict | None = None,
) -> None:
    """Инициализировать все singleton-сервисы один раз."""

    global _initialized
    if _initialized:
        return

    if redis_client is None:
        try:
            redis_client = redis.from_url(config.REDIS_URL)
            await redis_client.ping()
        except RedisError:
            redis_client = None
            logger.warning("Redis is unavailable; some features will stay disabled.")
        except Exception as exc:
            redis_client = None
            logger.error("Unexpected Redis initialization error: %s", exc)

    _services["redis"] = redis_client

    logger.info("Loading classification inference service")
    _services["inference"] = InferenceService(
        checkpoints_path=inference_checkpoint_path,
        idx_to_class=inference_idx_to_class or {},
    )

    logger.info("Loading image embedding service")
    _services["image_embedding"] = ImageEmbeddingService()

    provider = "onnx" if use_onnx else "torch"
    logger.info("Loading YOLO service with provider '%s'", provider)
    _services["yolo"] = YoloService(provider=provider)

    logger.info("Loading segmentation service")
    _services["segmentation"] = SegmentationService()

    logger.info("Loading NLP services")
    _services["embedding"] = EmbeddingService()
    _services["ner"] = NerService()
    _services["llm"] = LLMService()

    logger.info("Loading vector databases and recommenders")
    _services["vector_db"] = VectorDB(dim=_services["embedding"].dimension, redis_client=redis_client)
    _services["recsys_vector_db"] = RecSysVectorDB(dim=896, redis_client=redis_client)
    _services["content_based_recommender"] = ContentBasedRecommender(
        image_embedding_service=_services["image_embedding"],
        text_embedding_service=_services["embedding"],
        image_vector_db=_services["recsys_vector_db"],
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
    _services["rag"] = RAGService(
        llm_service=_services["llm"],
        semantic_search_service=_services["semantic_search"],
        redis=redis_client,
    )
    _services["drift_detector"] = DriftDetector()

    async with async_session() as session:
        tasks = await session.execute(select(Task).where(Task.avatar_file.isnot(None)))
        reference_embeddings = []
        for task in tasks.scalars():
            if task.avatar_file and os.path.exists(task.avatar_file):
                with open(task.avatar_file, "rb") as avatar_file:
                    reference_embeddings.append(_services["image_embedding"].get_embedding(avatar_file.read()))

        for embedding in reference_embeddings:
            _services["drift_detector"].add_embedding(embedding)

    _initialized = True
    logger.info("Services initialized successfully: %s", sorted(_services.keys()))


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
            inference_idx_to_class=inference_idx_to_class,
        )


def get_service(name: str) -> object:
    if name not in _services:
        raise RuntimeError(f"Service '{name}' is not initialized. Available: {list(_services.keys())}")
    return _services[name]


def get_redis() -> redis.Redis | None:
    return get_service("redis")


def get_inference() -> InferenceService:
    return get_service("inference")


def get_yolo() -> YoloService:
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
