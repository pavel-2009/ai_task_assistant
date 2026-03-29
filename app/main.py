"""
Точка входа в приложение.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
import sys
import os

from .ml.nlp.tasks import reindex_tasks
from .ml.recsys.tasks import train_collaborative_filtering_model

from app.db import close_redis, get_redis
from app.error_handlers import register_exception_handlers
from app.services import (
    init_services,
    get_embedding,
    get_ner,
    get_vector_db,
    get_semantic_search,
    get_llm,
    get_rag,
)
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = None

    try:
        logger.info("Connecting Redis...")
        redis_client = await get_redis()
        app.state.redis_client = redis_client
        logger.info("Redis connected successfully")

        logger.info("Initializing all services...")
        
        inference_checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "model.pth"
        inference_idx_to_class = {0: "cat", 1: "dog", 2: "house"}
        use_onnx = os.getenv("USE_ONNX", "False").lower() in ("true", "1", "t")
        
        await init_services(
            use_onnx=use_onnx,
            redis_client=redis_client,
            inference_checkpoint_path=str(inference_checkpoint_path),
            inference_idx_to_class=inference_idx_to_class,
        )
        logger.info("All services initialized successfully")

        # Сохраняем сервисы в app.state для доступа в роутерах
        app.state.embedding_service = get_embedding()
        app.state.ner_service = get_ner()
        app.state.vector_db = get_vector_db()
        app.state.semantic_search_service = get_semantic_search()
        app.state.llm_service = get_llm()
        app.state.rag_service = get_rag()
        
        logger.info("Starting background task for reindexing...")
        reindex_tasks.delay()
        logger.info("Background task for reindexing started successfully")
        
        logger.info("Starting background task for training collaborative filtering model...")
        train_collaborative_filtering_model.delay()
        logger.info("Background task for training collaborative filtering model started successfully")

    except Exception as exc:
        logger.error("Error during startup: %s", exc, exc_info=True)
        raise

    try:
        yield
    finally:
        await close_redis()


app = FastAPI(lifespan=lifespan)
register_exception_handlers(app)


from .routers import auth, avatars, nlp, rag, streaming, tasks

app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(avatars.router)
app.include_router(streaming.router)
app.include_router(nlp.router)
app.include_router(rag.router)


async def _get_model_health(app: FastAPI) -> dict[str, dict[str, object]]:
    embedding_service = getattr(app.state, "embedding_service", None)
    semantic_search_service = getattr(app.state, "semantic_search_service", None)
    ner_service = getattr(app.state, "ner_service", None)
    vector_db = getattr(app.state, "vector_db", None)
    
    OLLAMA_URL = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )
    OLLAMA_MODEL = os.getenv(
            "OLLAMA_MODEL",
            "llama3.2"
        )

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
            "model": OLLAMA_MODEL,
            "url": OLLAMA_URL
        },
        "rag": {
            "ready": rag_service is not None,
            "llm_model": OLLAMA_MODEL if rag_service is not None else None,
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
