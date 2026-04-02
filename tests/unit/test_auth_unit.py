"""Юнит-тесты auth-утилит c использованием in-memory SQLite."""

from datetime import timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import auth
from app.db_models import Base


TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="module")
async def auth_sessionmaker():
    engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker

    await engine.dispose()


@pytest.mark.unit
def test_create_access_token_has_user_id():
    token = auth.create_access_token(user_id=123, expires_delta=timedelta(minutes=5))
    payload = auth.jwt.decode(token, auth.config.SECRET_KEY, algorithms=[auth.config.JWT_ALGORITHM])

    assert payload["user_id"] == 123
    assert "exp" in payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_user_invalid_token(auth_sessionmaker):
    # Ключевой комментарий: проверяем только обработку ошибки декодирования JWT.
    async with auth_sessionmaker() as session:
        with pytest.raises(auth.HTTPException) as exc:
            await auth.get_current_user(token="broken.token", session=session)
        assert exc.value.status_code == 401
