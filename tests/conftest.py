from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    httpx = None
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("TEST_BASE_URL", "http://127.0.0.1:8000")

from app.core import config
from app.db import get_async_session
from app.db_models import Base
from app.error_handlers import register_exception_handlers
from app.routers import auth, nlp, rag, tasks
from tests.unit.mocks import (
    DummyDelayTask,
    DummyEmbeddingService,
    DummyNerService,
    DummyRagService,
    DummySemanticSearchService,
)

USER_A = {"username": "user_a", "password": "Password1!"}
USER_B = {"username": "user_b", "password": "Password1!"}


@pytest.fixture(scope="session")
def unit_db_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


# Изменяем scope на function, чтобы избежать конфликта с event_loop
@pytest.fixture(scope="function")
async def unit_session_maker(unit_db_engine):
    async with unit_db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(unit_db_engine, expire_on_commit=False)
    yield maker
    async with unit_db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await unit_db_engine.dispose()


@pytest.fixture(scope="function")
def unit_app(unit_session_maker) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth.router)
    app.include_router(tasks.router)
    app.include_router(nlp.router)
    app.include_router(rag.router)

    app.state.embedding_service = DummyEmbeddingService()
    app.state.semantic_search_service = DummySemanticSearchService()
    app.state.ner_service = DummyNerService()
    app.state.rag_service = DummyRagService()

    # Создаём синхронный генератор для сессии
    def _override_session() -> Generator[AsyncSession, None, None]:
        # Создаём новый event loop для каждого вызова
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _get_session():
            async with unit_session_maker() as session:
                return session
        
        session = loop.run_until_complete(_get_session())
        try:
            yield session
        finally:
            loop.run_until_complete(session.close())
            loop.close()
    
    app.dependency_overrides[get_async_session] = _override_session
    return app


@pytest.fixture(autouse=True)
def patch_celery_tasks(monkeypatch):
    dummy = DummyDelayTask()
    monkeypatch.setattr("app.routers.tasks.process_task_tags_and_embedding", dummy)
    monkeypatch.setattr("app.routers.tasks.process_task_interaction", dummy)
    monkeypatch.setattr("app.routers.tasks.update_recommendations_for_task", dummy)
    monkeypatch.setattr("app.routers.tasks.delete_task_interactions", dummy)
    monkeypatch.setattr("app.routers.rag.reindex_tasks_task", dummy)


@pytest.fixture(scope="function")
def unit_client_a(unit_app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(unit_app) as client:
        token = _register_and_login(client, USER_A)
        client.headers.update({"Authorization": f"Bearer {token}"})
        yield client


@pytest.fixture(scope="function")
def unit_client_b(unit_app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(unit_app) as client:
        token = _register_and_login(client, USER_B)
        client.headers.update({"Authorization": f"Bearer {token}"})
        yield client


@pytest.fixture(scope="session")
def integration_base_url() -> str:
    return config.TEST_BASE_URL


@pytest.fixture(scope="session")
def integration_httpx_client_a(integration_base_url: str):
    if httpx is None:
        pytest.skip("httpx is not installed")
    with httpx.Client(base_url=integration_base_url, timeout=20.0) as client:
        yield client


@pytest.fixture(scope="session")
def integration_httpx_client_b(integration_base_url: str):
    if httpx is None:
        pytest.skip("httpx is not installed")
    with httpx.Client(base_url=integration_base_url, timeout=20.0) as client:
        yield client


def _is_server_alive(client: httpx.Client) -> bool:
    try:
        return client.get("/ping").status_code in (200, 503)
    except Exception:
        return False


@pytest.fixture(scope="session")
def integration_token_a(integration_httpx_client_a: httpx.Client) -> str:
    if not _is_server_alive(integration_httpx_client_a):
        pytest.skip("Integration app is not running")
    return _register_and_login(integration_httpx_client_a, USER_A)


@pytest.fixture(scope="session")
def integration_token_b(integration_httpx_client_b: httpx.Client) -> str:
    if not _is_server_alive(integration_httpx_client_b):
        pytest.skip("Integration app is not running")
    return _register_and_login(integration_httpx_client_b, USER_B)


@pytest.fixture(scope="session")
def integration_auth_headers_a(integration_token_a: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {integration_token_a}"}


@pytest.fixture(scope="session")
def integration_auth_headers_b(integration_token_b: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {integration_token_b}"}


@pytest.fixture(scope="session")
def seeded_tasks_and_index(
    integration_httpx_client_a: httpx.Client,
    integration_httpx_client_b: httpx.Client,
    integration_auth_headers_a: dict[str, str],
    integration_auth_headers_b: dict[str, str],
):
    if not _is_server_alive(integration_httpx_client_a):
        pytest.skip("Integration app is not running")

    technologies = [
        "Python FastAPI Redis",
        "PostgreSQL SQLAlchemy Alembic",
        "Celery RabbitMQ Monitoring",
        "NLP embeddings vector search",
        "Docker CI/CD observability",
    ]

    created_ids = []
    for user_idx, (client, user_headers) in enumerate(((integration_httpx_client_a, integration_auth_headers_a), (integration_httpx_client_b, integration_auth_headers_b))):
        for idx in range(10):
            tech = technologies[idx % len(technologies)]
            payload = {
                "title": f"task-{user_idx}-{idx}-{tech.split()[0]}",
                "description": f"Implement feature with {tech} and add tests for NLP pipeline {idx}",
            }
            resp = client.post(
                "/tasks/", json=payload, headers=user_headers
            )
            assert resp.status_code in (200, 201), resp.text
            created_ids.append(resp.json()["id"])

            index_resp = integration_httpx_client_a.post("/nlp/index", json={"text": payload["description"]})
            assert index_resp.status_code == 200, index_resp.text

    reindex_resp = integration_httpx_client_a.post("/rag/reindex")
    assert reindex_resp.status_code == 200, reindex_resp.text

    return {"task_ids": created_ids, "count": len(created_ids)}


def _register_and_login(client, credentials: dict[str, str]) -> str:
    client.post("/auth/register", json=credentials)
    resp = client.post(
        "/auth/login",
        data={"username": credentials["username"], "password": credentials["password"]},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]
