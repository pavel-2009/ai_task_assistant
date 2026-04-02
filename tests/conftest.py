"""Базовые настройки и фикстуры для тестов."""

import pytest
import os

# Устанавливаем переменную окружения ПЕРЕД импортом конфига
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from fastapi.testclient import TestClient


# Телнологии в описаниях задач для тестирования
TECHNOLOGIES = [
    'Python', 'JavaScript', 'Java', 'C#', 'Ruby', 'Go', 'PHP', 'Swift', 'Kotlin', 'TypeScript'
]

# 10 фальшивых задач для тестирования
FAKE_TASKS = [
    {"title": f"Task {i}", "description": f"Description for task {i}: {TECHNOLOGIES[i % len(TECHNOLOGIES)]}"} for i in range(1, 11)
]


@pytest.fixture(scope="session", autouse=True)
def clean_metrics_registry():
    """Очищает реестр метрик перед запуском всех тестов, чтобы избежать дублирования."""
    from prometheus_client import REGISTRY
    
    # Удаляем все метрики перед тестами
    collectors_to_remove = list(REGISTRY._collector_to_names.keys())
    for collector in collectors_to_remove:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    
    yield


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Фикстура для создания всех таблиц в базе данных перед тестами."""
    from app.db import Base, engine
    import asyncio
    
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    asyncio.run(create_tables())
    
    yield


@pytest.fixture
def client():
    """Фикстура для создания тестового клиента."""
    from app import app
    return TestClient(app)



@pytest.fixture
def client2():
    """Фикстура для создания второго тестового клиента."""
    from app import app
    return TestClient(app)


@pytest.fixture
def fresh_app_client():
    """Фикстура для создания свежего клиента с новым экземпляром приложения."""
    import sys
    import importlib
    
    # Remove cached modules to force reimport
    modules_to_reload = [m for m in sys.modules if m.startswith('app')]
    for m in modules_to_reload:
        del sys.modules[m]
    
    # Re-import to get a fresh app instance
    from app import app
    return TestClient(app)


@pytest.fixture
def auth_token1(client, create_base_users):
    """Фикстура для получения токена аутентификации."""
    response = client.post("/auth/login", data={"username": "testuser", "password": "TestPass123!"})
    assert response.status_code == 200
    return response.json().get("access_token")


@pytest.fixture
def auth_token2(client, create_base_users):
    """Фикстура для получения второго токена аутентификации."""
    response = client.post("/auth/login", data={"username": "testuser2", "password": "TestPass456!"})
    assert response.status_code == 200
    return response.json().get("access_token")


@pytest.fixture()
def create_base_users(client):
    """Фикстура для создания базовых пользователей."""
    client.post("/auth/register", json={"username": "testuser", "password": "TestPass123!"})
    client.post("/auth/register", json={"username": "testuser2", "password": "TestPass456!"})


@pytest.fixture()
def create_base_users1(client):
    """Фикстура для создания базовых пользователей."""
    client.post("/auth/register", json={"username": "testuser", "password": "testpass"})
    
    
@pytest.fixture()
def create_base_users2(client2):
    """Фикстура для создания второго базового пользователя."""
    client2.post("/auth/register", json={"username": "testuser2", "password": "testpass"})


@pytest.fixture()
def create_base_tasks(client, auth_token):
    """Фикстура для создания базовых задач."""
    for task in FAKE_TASKS:
        response = client.post("/tasks/", data=task, headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        
        
@pytest.fixture()
def create_base_tasks_for_user2(client, auth_token2):
    """Фикстура для создания базовых задач для второго пользователя."""
    for task in FAKE_TASKS:
        response = client.post("/tasks/", json=task, headers={"Authorization": f"Bearer {auth_token2}"})
        assert response.status_code == 200
        
        
@pytest.fixture()
def engine():
    """Фикстура для получения движка базы данных."""
    from app.db import engine
    return engine


@pytest.fixture()
async def session(engine):
    """Фикстура для получения сессии базы данных."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async with async_session() as session:
        yield session