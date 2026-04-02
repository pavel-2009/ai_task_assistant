"""Общие фикстуры для API-тестов с SQLite test.db в корне проекта."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db_models import Base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DB_PATH = PROJECT_ROOT / "test.db"
TEST_DB_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH}"


@pytest.fixture(scope="session")
async def api_sessionmaker():
    """Создаем test.db и таблицы один раз на сессию."""
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    yield session_maker

    await engine.dispose()
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
