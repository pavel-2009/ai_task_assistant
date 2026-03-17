"""
Точка входа в приложение
"""

from fastapi import FastAPI, status

from contextlib import asynccontextmanager
import logging

from app.celery_app import preload_models
from app.ml.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # === STARTUP ===
    
    try:
        logger.info("Preloading models...")
        await preload_models()
        logger.info("Models preloaded successfully")
        
        logger.info("Initializing EmbeddingService...")
        embedding_service = EmbeddingService()
        app.state.embedding_service = embedding_service
        app.state.embedding_service.encode("Test embedding to warm up the model")  # Прогрев модели
        logger.info("EmbeddingService initialized successfully")
        
        print("App started")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        print(f"Error: {e}")
     
    yield
            
    # === SHUTDOWN ===
            

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
