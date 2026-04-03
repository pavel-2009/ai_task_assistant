"""Юнит-тесты auth-утилит c использованием in-memory SQLite."""

from datetime import timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import auth
from app.db_models import Base


TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_user_valid_token(auth_sessionmaker):
    """Тестирование получения пользователя по валидному токену."""
    from app.db_models import User
    
    async with auth_sessionmaker() as session:
        # Создаем пользователя в БД
        user = User(username="testuser", password="hashed_password")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        # Создаем токен для этого пользователя
        token = auth.create_access_token(user_id=user.id)
        
        # Получаем пользователя по токену
        result_user = await auth.get_current_user(token=token, session=session)
        
        assert result_user.id == user.id
        assert result_user.username == "testuser"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_user_nonexistent_user(auth_sessionmaker):
    """Тестирование получения несуществующего пользователя."""
    async with auth_sessionmaker() as session:
        # Создаем токен для несуществующего пользователя
        token = auth.create_access_token(user_id=9999)
        
        # Должно вернуть ошибку
        with pytest.raises(auth.HTTPException) as exc:
            await auth.get_current_user(token=token, session=session)
        assert exc.value.status_code == 401
