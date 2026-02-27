from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from typing import AsyncGenerator


DATABASE_URL = "sqlite+aiosqlite:///./test.db"

Base = declarative_base()

engine = create_async_engine(DATABASE_URL)

async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Создает асинхронную сессию для работы с базой данных."""

    async with async_session() as session:
        yield session
