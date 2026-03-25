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

from app.celery_app import preload_models
from app.db import close_redis, get_redis
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.vector_db import VectorDB
from app.ml.nlp.ner_service import NerService
from app.ml.nlp.llm_service import LLMService
from app.ml.nlp.rag_service import RAGService

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = None

    try:
        logger.info("Preloading models...")
        await preload_models()
        logger.info("Models preloaded successfully")

        logger.info("Connecting Redis for NLP...")
        redis_client = await get_redis()
        app.state.redis_client = redis_client
        logger.info("Redis connected successfully")

        logger.info("Initializing EmbeddingService...")
        embedding_service = EmbeddingService()
        app.state.embedding_service = embedding_service
        embedding_service.encode_one("Test embedding to warm up the model")
        logger.info("EmbeddingService initialized successfully")

        logger.info("Loading FAISS database...")
        vector_db = VectorDB(dim=embedding_service.dimension, redis_client=redis_client)
        await vector_db.load_from_redis()
        app.state.vector_db = vector_db
        logger.info("FAISS database loaded successfully")

        logger.info("Loading SemanticSearchService...")
        semantic_search_service = SemanticSearchService(
            embedding_service=embedding_service,
            vector_db=vector_db,
            redis_client=redis_client,
        )
        app.state.semantic_search_service = semantic_search_service
        logger.info("SemanticSearchService loaded successfully")

        logger.info("Loading NER model...")
        ner_service = NerService()
        app.state.ner_service = ner_service
        logger.info("NER model loaded successfully")
        
        logger.info("Initializing LLMService...")
        llm_service = LLMService()
        app.state.llm_service = llm_service

        logger.info("Initializing RAGService...")
        rag_service = RAGService(
            llm_service=llm_service,
            semantic_search_service=semantic_search_service,
            redis=redis_client,
        )
        app.state.rag_service = rag_service
        logger.info("RAGService initialized successfully")
        
        logger.info("Starting background task for reindexing...")
        reindex_tasks.delay()
        logger.info("Background task for reindexing started successfully")

    except Exception as exc:
        logger.error("Error during startup: %s", exc, exc_info=True)
        raise

    try:
        yield
    finally:
        await close_redis()


app = FastAPI(lifespan=lifespan)


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
