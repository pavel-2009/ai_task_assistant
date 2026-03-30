"""
Точка входа в приложение.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status

from app.core import config
from app.db import close_redis, engine
from app.error_handlers import register_exception_handlers
from app.celery_app import celery_app
from app.services import (
    get_embedding,
    get_llm,
    get_ner,
    get_rag,
    get_semantic_search,
    get_vector_db,
    ensure_services_initialized,
    get_drift_detector,
    get_redis,
)

from .ml.nlp.tasks import reindex_tasks
from .ml.recsys.tasks import train_collaborative_filtering_model

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Переменные для хранения ID фоновых задач
_background_tasks = {
    "reindex_tasks": None,
    "train_cf": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Initializing all services...")
        await ensure_services_initialized(
            use_onnx=config.USE_ONNX,
        )
        logger.info("All services initialized successfully")

        app.state.embedding_service = get_embedding()
        app.state.ner_service = get_ner()
        app.state.vector_db = get_vector_db()
        app.state.semantic_search_service = get_semantic_search()
        app.state.llm_service = get_llm()
        app.state.rag_service = get_rag()
        app.state.drift_detector = get_drift_detector()
        app.state.redis_client = get_redis()

        # Запуск warmup LLM в фоне через Celery (не блокирует запуск API)
        try:
            from app.ml.nlp.tasks import warmup_llm
            warmup_llm.delay()
            logger.info("LLM warmup task scheduled")
        except Exception as warmup_exc:
            logger.warning("Could not schedule LLM warmup task: %s", warmup_exc)

        # Запуск фоновых задач (может быть недоступно в тестовом окружении)
        try:
            logger.info("Starting background task for reindexing...")
            _background_tasks["reindex_tasks"] = reindex_tasks.delay()
            logger.info("Background task for reindexing started successfully")

            logger.info("Starting background task for training collaborative filtering model...")
            _background_tasks["train_cf"] = train_collaborative_filtering_model.delay()
            logger.info("Background task for training collaborative filtering model started successfully")
        except Exception as bg_exc:
            logger.warning("Could not start background tasks (Redis may be unavailable): %s", bg_exc)
            # Не прерываем запуск приложения, продолжаем работу без фоновых задач

    except Exception as exc:
        logger.error("Error during startup: %s", exc, exc_info=True)
        raise

    try:
        yield
    finally:
        logger.info("Initiating graceful shutdown...")
        
        # Отменяем все активные фоновые задачи
        logger.info("Отмена фоновых задач...")
        for task_name, task in _background_tasks.items():
            if task:
                try:
                    task.revoke(terminate=False)
                    logger.info(f"Задача '{task_name}' отменена")
                except Exception as e:
                    logger.warning(f"Ошибка при отмене задачи '{task_name}': {e}")
        
        # Даем Celery задачам 30 секунд на завершение
        logger.info("Ожидаю завершения Celery задач (30 сек)...")
        try:
            # Отправляем сигнал graceful shutdown в Celery workers
            celery_app.control.shutdown(timeout=30)
            time.sleep(2)  # Даем время на обработку
        except Exception as e:
            logger.warning(f"Ошибка при shutdown Celery: {e}")
        
        # Закрываем Redis
        logger.info("Закрытие соединения с Redis...")
        try:
            await close_redis()
            logger.info("Redis соединение закрыто")
        except Exception as e:
            logger.warning(f"Ошибка при закрытии Redis: {e}")
        
        # Закрываем async_session и engine
        logger.info("Закрытие database engine...")
        try:
            await engine.dispose()
            logger.info("Database engine закрыт")
        except Exception as e:
            logger.warning(f"Ошибка при закрытии engine: {e}")
        
        logger.info("Graceful shutdown завершен")


app = FastAPI(lifespan=lifespan)
register_exception_handlers(app)


from .routers import auth, avatars, nlp, rag, streaming, tasks, monitoring

app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(avatars.router)
app.include_router(streaming.router)
app.include_router(nlp.router)
app.include_router(rag.router)
app.include_router(monitoring.router)


async def _get_model_health(app: FastAPI) -> dict[str, dict[str, object]]:
    embedding_service = getattr(app.state, "embedding_service", None)
    semantic_search_service = getattr(app.state, "semantic_search_service", None)
    ner_service = getattr(app.state, "ner_service", None)
    vector_db = getattr(app.state, "vector_db", None)

    embedding_ready = bool(
        embedding_service is not None
        and getattr(embedding_service, "model", None) is not None
        and getattr(embedding_service, "dimension", 0) > 0
    )
    vector_db_ready = bool(
        vector_db is not None
        and getattr(vector_db, "index", None) is not None
        and getattr(vector_db, "dim", 0) > 0
    )
    semantic_search_ready = bool(
        semantic_search_service is not None
        and getattr(semantic_search_service, "embedding_service", None) is not None
        and getattr(semantic_search_service, "vector_db", None) is not None
    )
    ner_ready = bool(ner_service is not None and ner_service.is_ready)
    llm_service = getattr(app.state, "llm_service", None)
    rag_service = getattr(app.state, "rag_service", None)
    
    drift_detector = getattr(app.state, "drift_detector", None)
    drift_status = drift_detector.get_status() if drift_detector else {}


    return {
        "embedding": {
            "ready": embedding_ready,
            "model": "sentence-transformers/all-MiniLM-L6-v2" if embedding_service is not None else None,
            "dimension": getattr(embedding_service, "dimension", None),
        },
        "vector_db": {
            "ready": vector_db_ready,
            "dimension": getattr(vector_db, "dim", None),
            "indexed_items": len(getattr(vector_db, "ids", [])) if vector_db is not None else None,
        },
        "semantic_search": {
            "ready": semantic_search_ready,
            "cache_prefix": getattr(semantic_search_service, "cache_prefix", None),
        },
        "ner": {
            "ready": ner_ready,
            "model": "en_core_web_sm" if ner_service is not None else None,
        },
        "llm": {
            "ready": llm_service and await llm_service.is_available(),
            "model": config.OLLAMA_MODEL,
            "url": config.OLLAMA_BASE_URL,
        },
        "rag": {
            "ready": rag_service is not None,
            "llm_model": config.OLLAMA_MODEL if rag_service is not None else None,
        },
        "data_drift": {
            "ready": drift_detector is not None,
            "drift_detected": drift_status.get("drift_detected", False),
            "drift_score": drift_status.get("drift_score", 0.0),
        }
    }


@app.get("/ping", status_code=status.HTTP_200_OK, description="Health-check endpoint")
async def ping(request: Request, response: Response):
    """Информация о состоянии приложения и подключенных моделей"""
    models = await _get_model_health(request.app)
    all_models_ready = all(component["ready"] for component in models.values())

    if not all_models_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if all_models_ready else "degraded",
        "models_ready": all_models_ready,
        "models": models,
    }
