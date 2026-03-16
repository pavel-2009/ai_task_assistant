"""
Точка входа в приложение
"""

from fastapi import FastAPI, status

from contextlib import asynccontextmanager

from app.celery_app import preload_models


# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # === STARTUP ===
    
    try:
        await preload_models()
        print("App started")
        
    except Exception as e:
        print(f"Error: {e}")
     
    yield
            
    # === SHUTDOWN ===
            

# Создание приложения
app = FastAPI(lifespan=lifespan)


# Подключение роутеров
from .routers import auth, tasks, avatars, streaming

app.include_router(auth.router)
app.include_router(tasks.router)
app.include_router(avatars.router)
app.include_router(streaming.router)


# === ЭНДПОИНТ ПРОВЕРКИ ЗДОРОВЬЯ ===
@app.get("/ping", status_code=status.HTTP_200_OK, description="Health-check эндпоинт")
async def ping():
    """Health-check"""

    return {
        "status": "OK"
    }
