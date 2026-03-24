from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from typing import AsyncGenerator
import redis.asyncio as redis
import os


DATABASE_URL = "sqlite+aiosqlite:///./test.db"

Base = declarative_base()

engine = create_async_engine(DATABASE_URL)

async_session = async_sessionmaker(engine, expire_on_commit=False)


# Redis для кэширования и фоновых задач
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/1")
_redis_client = None


async def get_redis() -> redis.Redis:
    """Получить Redis клиент (синглтон)"""
    global _redis_client
    if _redis_client is None:
        _redis_client = await redis.from_url(REDIS_URL)
    return _redis_client


async def close_redis():
    """Закрыть Redis соединение"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Создает асинхронную сессию для работы с базой данных."""

    async with async_session() as session:
        yield session
