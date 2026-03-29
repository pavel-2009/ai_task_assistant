from typing import AsyncGenerator

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core import config


Base = declarative_base()

engine = create_async_engine(config.DATABASE_URL)
async_session = async_sessionmaker(engine, expire_on_commit=False)

_redis_client: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Получить Redis клиент (синглтон)"""
    global _redis_client
    if _redis_client is None:
        _redis_client = await redis.from_url(config.REDIS_URL)
    return _redis_client


async def close_redis() -> None:
    """Закрыть Redis соединение"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Создает асинхронную сессию для работы с базой данных."""

    async with async_session() as session:
        yield session
