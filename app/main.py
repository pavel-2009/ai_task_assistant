"""
Точка входа в приложение.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from redis import Redis

from app.celery_app import preload_models
from app.db import REDIS_URL, close_redis
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.vector_db import VectorDB

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client: Redis | None = None

    try:
        logger.info("Preloading models...")
        await preload_models()
        logger.info("Models preloaded successfully")

        logger.info("Connecting Redis for NLP...")
        redis_client = Redis.from_url(REDIS_URL, decode_responses=False)
        app.state.redis_client = redis_client
        logger.info("Redis connected successfully")

        logger.info("Initializing EmbeddingService...")
        embedding_service = EmbeddingService()
        app.state.embedding_service = embedding_service
        embedding_service.encode_one("Test embedding to warm up the model")
        logger.info("EmbeddingService initialized successfully")

        logger.info("Loading FAISS database...")
        vector_db = VectorDB(dim=embedding_service.dimension, redis_client=redis_client)
        vector_db.load_from_redis()
        app.state.vector_db = vector_db
        logger.info("FAISS database loaded successfully")

        logger.info("Loading SemanticSearchService...")
        semantic_search_service = SemanticSearchService(
            embedding_service=embedding_service,
            database=vector_db,
            redis_client=redis_client,
        )
        app.state.semantic_search_service = semantic_search_service
        logger.info("SemanticSearchService loaded successfully")
    except Exception as exc:
        logger.error("Error during startup: %s", exc, exc_info=True)
        raise

    try:
        yield
    finally:
        if redis_client is not None:
            redis_client.close()
        await close_redis()


app = FastAPI(lifespan=lifespan)


from .routers import auth, avatars, nlp, streaming, tasks

app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(avatars.router)
app.include_router(streaming.router)
app.include_router(nlp.router)


@app.get("/ping", status_code=status.HTTP_200_OK, description="Health-check эндпоинт")
async def ping():
    """Health-check."""
    return {"status": "OK"}
