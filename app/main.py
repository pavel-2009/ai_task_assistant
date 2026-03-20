"""
Точка входа в приложение
"""

from fastapi import FastAPI, status

from contextlib import asynccontextmanager
import logging

from app.celery_app import preload_models
from app.db import close_redis, get_redis
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.vector_db import VectorDB

logger = logging.getLogger(__name__)


# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # === STARTUP ===
    
    try:
        logger.info("Preloading models...")
        await preload_models()
        logger.info("Models preloaded successfully")
        
        logger.info("Connecting Redis for NLP services...")
        redis_client = get_redis()
        app.state.redis_client = redis_client

        logger.info("Loading FAISS database...")
        vector_db = VectorDB(dim=384, redis_client=redis_client)  # Инициализируем базу данных с размерностью эмбеддингов модели
        app.state.vector_db = vector_db
        logger.info("FAISS database loaded successfully")
        
        logger.info("Initializing EmbeddingService...")
        embedding_service = EmbeddingService()
        app.state.embedding_service = embedding_service
        app.state.embedding_service.encode_one("Test embedding to warm up the model")  # Прогрев модели
        logger.info("EmbeddingService initialized successfully")
        
        logger.info("Loading SemanticSearchService...")
        semantic_search_service = SemanticSearchService(app.state.embedding_service, app.state.vector_db, redis_client=redis_client)
        app.state.semantic_search_service = semantic_search_service
        logger.info("SemanticSearchService loaded successfully")
        
        print("App started")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        print(f"Error: {e}")
     
    yield

    # === SHUTDOWN ===
    close_redis()
            

# Создание приложения
app = FastAPI(lifespan=lifespan)


# Подключение роутеров
from .routers import auth, tasks, avatars, streaming, nlp

app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(avatars.router)
app.include_router(streaming.router)
app.include_router(nlp.router)


# === ЭНДПОИНТ ПРОВЕРКИ ЗДОРОВЬЯ ===
@app.get("/ping", status_code=status.HTTP_200_OK, description="Health-check эндпоинт")
async def ping():
    """Health-check"""

    return {
        "status": "OK"
    }
