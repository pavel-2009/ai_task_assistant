"""
Глобальный реестр сервисов. Инициализируется в lifespan приложения.
Доступен как FastAPI, так и Celery воркерам.
"""

from __future__ import annotations

from app.ml.cv.classification.inference_service import InferenceService
from app.ml.cv.detection.yolo_service import YoloService
from app.ml.cv.detection.yolo_onnx_service import YoloONNXService
from app.ml.cv.segmentation.segmentation_service import SegmentationService
from app.ml.nlp.rag_service import RAGService
from app.ml.nlp.vector_db import VectorDB
from app.ml.nlp.ner_service import NerService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.llm_service import LLMService

import redis.asyncio as redis


_services: dict = {}


async def init_services(
    use_onnx: bool = False,
    redis_client: redis.Redis | None = None,
    inference_checkpoint_path: str = None,
    inference_idx_to_class: dict = None,
) -> None:
    """Инициализирует все сервисы один раз при старте приложения."""
    
    # CV сервисы
    _services["inference"] = InferenceService(
        checkpoint_path=inference_checkpoint_path,
        idx_to_class=inference_idx_to_class or {}
    )
    
    if use_onnx:
        _services["yolo"] = YoloONNXService()
    else:
        _services["yolo"] = YoloService()
    
    _services["segmentation"] = SegmentationService()
    
    # NLP сервисы
    _services["embedding"] = EmbeddingService()
    _services["ner"] = NerService()
    _services["llm"] = LLMService()
    
    # Vector DB и семантический поиск
    _services["vector_db"] = VectorDB(
        dim=_services["embedding"].dimension,
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


def get_service(name: str) -> object:
    """Получить инициализированный сервис."""
    
    if name not in _services:
        raise RuntimeError(
            f"Сервис '{name}' не инициализирован. "
            f"Доступные сервисы: {list(_services.keys())}"
        )
    return _services[name]


def get_inference() -> InferenceService:
    return get_service("inference")


def get_yolo() -> YoloService | YoloONNXService:
    return get_service("yolo")


def get_segmentation() -> SegmentationService:
    return get_service("segmentation")


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
