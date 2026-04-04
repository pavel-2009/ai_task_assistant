"""Базовые настройки и фикстуры для тестов."""

import pytest
import os
import asyncio
import time
import requests

# Устанавливаем переменную окружения ПЕРЕД импортом конфига
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("JWT_EXPIRE_MINUTES", "30")

# Импортируем app СРАЗУ для инициализации engine с тестовой БД
from app import app  # noqa
from fastapi.testclient import TestClient


# Технологии в описаниях задач для тестирования
TECHNOLOGIES = [
    'Python', 'JavaScript', 'Java', 'C#', 'Ruby', 'Go', 'PHP', 'Swift', 'Kotlin', 'TypeScript'
]

# 10 фальшивых задач для тестирования
FAKE_TASKS = [
    {"title": f"Task {i}", "description": f"Description for task {i}: {TECHNOLOGIES[i % len(TECHNOLOGIES)]}"} for i in range(1, 11)
]

# 20 фальшивых задач для тестирования (10 для каждого пользователя)
FAKE_TASKS_USER1 = [
    {"title": f"Task {i}", "description": f"Description for task {i}: {TECHNOLOGIES[i % len(TECHNOLOGIES)]}"} for i in range(1, 11)
]
FAKE_TASKS_USER2 = [
    {"title": f"Task {i}", "description": f"Description for task {i}: {TECHNOLOGIES[i % len(TECHNOLOGIES)]}"} for i in range(11, 21)
]


@pytest.fixture(scope="session", autouse=True)
def wait_for_services():
    """Ждем пока все сервисы запустятся (RAG, LLM, Embedding)."""
    max_attempts = 90
    for attempt in range(max_attempts):
        try:
            # Пробуем подключиться к реальному серверу (для CI)
            resp = requests.get("http://localhost:8000/ping", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", {})
                
                # Проверяем готовность критических сервисов
                rag_ready = models.get("rag", {}).get("ready", False)
                llm_ready = models.get("llm", {}).get("ready", False)
                embedding_ready = models.get("embedding", {}).get("ready", False)
                ner_ready = models.get("ner", {}).get("ready", False)
                
                if rag_ready and llm_ready and embedding_ready and ner_ready:
                    print(f"✅ All services ready! (attempt {attempt + 1})")
                    return
                
                print(f"⏳ Waiting for services (attempt {attempt + 1}): "
                      f"rag={rag_ready}, llm={llm_ready}, "
                      f"embedding={embedding_ready}, ner={ner_ready}")
        except requests.exceptions.ConnectionError:
            print(f"⏳ Connection refused (attempt {attempt + 1}/{max_attempts})")
        except Exception as e:
            print(f"⚠️ Error checking services: {e}")
        
        time.sleep(2)
    
    print("❌ WARNING: Services not fully ready after timeout, but continuing tests...")


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
    """Фикстура для создания всех таблиц в базе данных перед тестами - ОДИН РАЗ для всей сессии."""
    from app.db import Base, engine
    
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    asyncio.run(create_tables())
    
    yield


@pytest.fixture(autouse=True)
def cleanup_tasks_between_tests():
    """Очищает задачи перед каждым тестом для изоляции."""
    # ДО теста - очищаем задачи (но не пользователей)
    from app.db import engine
    from app.db_models import Task
    from sqlalchemy import delete
    
    async def clear_tasks():
        async with engine.begin() as conn:
            await conn.execute(delete(Task))
            await conn.commit()
    
    asyncio.run(clear_tasks())
    
    yield  # Выполняем тест


@pytest.fixture(scope="session")
def init_app_services():
    """Инициализирует NLP/RAG сервисы и сохраняет их в app.state для TestClient."""
    from app.services import (
        get_embedding,
        get_ner,
        get_rag,
        get_semantic_search,
        init_services,
    )

    async def _init():
        await init_services(use_onnx=False)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        temp_loop = asyncio.new_event_loop()
        try:
            temp_loop.run_until_complete(_init())
        finally:
            temp_loop.close()
    else:
        loop.run_until_complete(_init())

    app.state.embedding_service = get_embedding()
    app.state.semantic_search_service = get_semantic_search()
    app.state.ner_service = get_ner()
    app.state.rag_service = get_rag()

    return app


@pytest.fixture
def client(init_app_services):
    """Фикстура для создания тестового клиента."""
    return TestClient(app)


@pytest.fixture
def client2(init_app_services):
    """Фикстура для создания второго тестового клиента."""
    from app import app
    return TestClient(app)


@pytest.fixture(scope="session")
def create_base_users_session(init_app_services):
    """Фикстура для создания базовых пользователей (один раз на сессию)."""
    client = TestClient(app)
    client.post("/auth/register", json={"username": "testuser", "password": "TestPass123!"})
    client.post("/auth/register", json={"username": "testuser2", "password": "TestPass456!"})
    return client


@pytest.fixture(scope="session")
def auth_token_session(create_base_users_session):
    """Фикстура для получения токена первого пользователя (один раз на сессию)."""
    response = create_base_users_session.post("/auth/login", data={"username": "testuser", "password": "TestPass123!"})
    assert response.status_code == 200
    return response.json().get("access_token")


@pytest.fixture(scope="session")
def auth_token2_session(create_base_users_session):
    """Фикстура для получения токена второго пользователя (один раз на сессию)."""
    response = create_base_users_session.post("/auth/login", data={"username": "testuser2", "password": "TestPass456!"})
    assert response.status_code == 200
    return response.json().get("access_token")


@pytest.fixture
def auth_token(auth_token_session):
    """Фикстура для получения токена (переиспользует session токен)."""
    return auth_token_session


@pytest.fixture
def auth_token1(auth_token_session):
    """Фикстура для получения токена аутентификации (переиспользует session токен)."""
    return auth_token_session


@pytest.fixture
def auth_token2(auth_token2_session):
    """Фикстура для получения второго токена аутентификации (переиспользует session токен)."""
    return auth_token2_session


@pytest.fixture
def authorized_client(client, auth_token):
    """Фикстура для создания авторизованного клиента."""
    client.headers.update({"Authorization": f"Bearer {auth_token}"})
    return client


@pytest.fixture
def authorized_client2(client2, auth_token2):
    """Фикстура для создания второго авторизованного клиента."""
    client2.headers.update({"Authorization": f"Bearer {auth_token2}"})
    return client2


@pytest.fixture
def create_tasks_for_auth_client2(authorized_client2):
    """Фикстура для создания задач для второго авторизованного клиента."""
    for task in FAKE_TASKS:
        response = authorized_client2.post("/tasks/", json=task)
        assert response.status_code == 201


@pytest.fixture
def fresh_app_client(init_app_services):
    """Фикстура для создания свежего клиента с новым экземпляром приложения."""
    import sys
    
    # Remove cached modules to force reimport
    modules_to_reload = [m for m in sys.modules if m.startswith('app')]
    for m in modules_to_reload:
        del sys.modules[m]
    
    # Re-import to get a fresh app instance
    from app import app as fresh_app
    from app.db import Base, engine as fresh_engine
    
    # Создаём таблицы в новом engine
    async def create_tables():
        async with fresh_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    asyncio.run(create_tables())
    
    return TestClient(fresh_app)


@pytest.fixture()
def create_base_users(create_base_users_session):
    """Фикстура для совместимости (переиспользует session версию)."""
    return create_base_users_session


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
    """Фикстура для создания базовых задач (1-10) для первого пользователя."""
    for task in FAKE_TASKS_USER1:
        response = client.post("/tasks/", json=task, headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 201
        
        
@pytest.fixture()
def create_base_tasks_for_user2(client, auth_token2):
    """Фикстура для создания базовых задач (11-20) для второго пользователя."""
    for task in FAKE_TASKS_USER2:
        response = client.post("/tasks/", json=task, headers={"Authorization": f"Bearer {auth_token2}"})
        assert response.status_code == 201


@pytest.fixture()
def create_all_base_tasks(client, auth_token, auth_token2):
    """Фикстура для создания всех базовых задач (1-10 для user1, 11-20 для user2)."""
    # Создаем задачи 1-10 для первого пользователя
    for task in FAKE_TASKS_USER1:
        response = client.post("/tasks/", json=task, headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 201
    
    # Создаем задачи 11-20 для второго пользователя
    for task in FAKE_TASKS_USER2:
        response = client.post("/tasks/", json=task, headers={"Authorization": f"Bearer {auth_token2}"})
        assert response.status_code == 201
        
        
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
        
        
# === Фикстуры для RAG API ===
@pytest.fixture
def reindex_tasks(authorized_client):
    """Фикстура для переиндексации задач в RAG API перед тестами."""
    # Ждем готовности сервисов перед реиндексацией
    max_retries = 30
    for attempt in range(max_retries):
        try:
            health_response = authorized_client.get("/ping")
            if health_response.status_code == 200:
                data = health_response.json()
                if data.get("models", {}).get("rag", {}).get("ready", False):
                    break
        except:
            pass
        time.sleep(2)
    
    response = authorized_client.post("/rag/reindex")
    assert response.status_code == 200
    
    # Даем время на индексацию
    time.sleep(3)