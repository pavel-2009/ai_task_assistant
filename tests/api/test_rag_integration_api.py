"""Интеграционные тесты auth/tasks/rag с глобальной инициализацией сервисов и Celery reindex."""

from __future__ import annotations

import uuid

import faiss
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

import app.services as services_registry
from app.db import async_session
from app.db_models import Task, Text, User
from app.error_handlers import register_exception_handlers
from app.ml.nlp.tasks import reindex_tasks
from app.routers import auth as auth_router
from app.routers import nlp as nlp_router
from app.routers import rag as rag_router
from app.routers import tasks as tasks_router


TECH_TASKS = [
    ("Python API", "Build backend with Python and FastAPI"),
    ("JavaScript UI", "Create frontend with JavaScript and React"),
    ("Java Service", "Develop microservice on Java and Spring"),
    ("C# Core", "Refactor legacy C# and .NET module"),
    ("Go ETL", "Implement ETL pipeline in Go"),
    ("Kotlin App", "Add Android module on Kotlin"),
    ("Swift iOS", "Deliver iOS feature with Swift"),
    ("TS SDK", "Ship TypeScript SDK for API"),
    ("Rust Worker", "Create high-load worker on Rust"),
    ("PHP Mig", "Migrate old PHP endpoints"),
    ("Ruby Script", "Automate reporting with Ruby"),
    ("SQL Tuning", "Tune PostgreSQL queries"),
    ("K8s Ops", "Deploy workloads in Kubernetes"),
    ("Terraform", "Manage IaC in Terraform"),
    ("ML Serve", "Serve PyTorch inference model"),
]


@pytest_asyncio.fixture(scope="module")
async def initialized_services():
    """Только глобальная инициализация сервисов (как для Celery)."""
    await services_registry.ensure_services_initialized()
    return {
        "embedding": services_registry.get_embedding(),
        "semantic": services_registry.get_semantic_search(),
        "llm": services_registry.get_llm(),
        "rag": services_registry.get_rag(),
    }


@pytest.fixture(scope="module")
def stage_client(initialized_services):
    """Единый API клиент для этапов auth/tasks/rag."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router.router)
    app.include_router(tasks_router.router)
    app.include_router(nlp_router.router)
    app.include_router(rag_router.router)

    app.state.embedding_service = initialized_services["embedding"]
    app.state.semantic_search_service = initialized_services["semantic"]
    app.state.llm_service = initialized_services["llm"]
    app.state.rag_service = initialized_services["rag"]

    client = TestClient(app)
    yield client
    client.close()


@pytest_asyncio.fixture()
async def auth_stage_data():
    """Очистка/подготовка БД перед auth-этапом."""
    async with async_session() as session:
        await session.execute(delete(User).where(User.username.like("stage_auth_%")))
        await session.commit()


@pytest_asyncio.fixture()
async def tasks_stage_data():
    """Очистка/подготовка БД перед tasks-этапом."""
    async with async_session() as session:
        await session.execute(delete(Text))
        await session.execute(delete(Task))
        await session.commit()


@pytest_asyncio.fixture()
async def rag_stage_data(tasks_stage_data, initialized_services):
    """Очистка/наполнение БД перед RAG-этапом + отдельный вызов Celery reindex task."""
    semantic = initialized_services["semantic"]
    vector_db = semantic.vector_db

    async with async_session() as session:
        username = f"stage_auth_{uuid.uuid4().hex[:8]}"
        user = User(username=username, password="hashed_password")
        session.add(user)
        await session.flush()

        for title, description in TECH_TASKS:
            session.add(Task(title=title, description=description, author_id=user.id))

        await session.commit()

    # Перед reindex очищаем индекс, затем вызываем reindex task ОТДЕЛЬНО
    vector_db.index = faiss.IndexFlatIP(vector_db.dim)
    vector_db.ids.clear()
    vector_db.ids_to_idx.clear()
    reindex_tasks()


@pytest.mark.asyncio
async def test_auth_stage_reset_and_register(stage_client, auth_stage_data):
    username = f"stage_auth_{uuid.uuid4().hex[:8]}"
    response = stage_client.post("/auth/register", json={"username": username, "password": "StrongPass123!"})
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_tasks_stage_db_reset(tasks_stage_data):
    async with async_session() as session:
        tasks_count = len((await session.execute(select(Task.id))).all())
        texts_count = len((await session.execute(select(Text.id))).all())
    assert tasks_count == 0
    assert texts_count == 0


@pytest.mark.asyncio
async def test_semantic_reindex_via_celery_task(rag_stage_data, initialized_services):
    vector_db = initialized_services["semantic"].vector_db
    assert len(vector_db.ids) >= len(TECH_TASKS)


@pytest.mark.asyncio
async def test_rag_ask_edge_cases(stage_client, rag_stage_data):
    ok_resp = stage_client.post(
        "/rag/ask",
        json={
            "query": "Какие задачи есть по Python и FastAPI?",
            "top_k": 5,
            "use_cache": False,
            "filename": "q.txt",
            "size": 10,
        },
    )
    assert ok_resp.status_code == 200
    ok_payload = ok_resp.json()
    assert isinstance(ok_payload.get("answer"), str) and ok_payload["answer"]
    assert ok_payload.get("sources")

    empty_query = stage_client.post(
        "/rag/ask",
        json={"query": "", "top_k": 5, "use_cache": False, "filename": "e.txt", "size": 1},
    )
    assert empty_query.status_code == 422

    invalid_top_k = stage_client.post(
        "/rag/ask",
        json={"query": "python", "top_k": 50, "use_cache": False, "filename": "t.txt", "size": 1},
    )
    assert invalid_top_k.status_code == 422


@pytest.mark.asyncio
async def test_rag_repeat_question_cache_behavior(stage_client, rag_stage_data):
    payload = {
        "query": "Что есть по Kubernetes?",
        "top_k": 3,
        "use_cache": True,
        "filename": "k8s.txt",
        "size": 2,
    }

    first = stage_client.post("/rag/ask", json=payload)
    second = stage_client.post("/rag/ask", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200


@pytest.mark.asyncio
async def test_rag_stream_and_internal_errors(stage_client, rag_stage_data, initialized_services):
    with stage_client.stream(
        "POST",
        "/rag/ask/stream",
        json={"query": "TypeScript", "top_k": 3, "use_cache": False, "filename": "s.txt", "size": 1},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        stream_data = "".join(stream_resp.iter_text())
    assert "event: done" in stream_data

    rag_service = initialized_services["rag"]
    original_semantic = rag_service.semantic_search_service
    rag_service.semantic_search_service = None
    try:
        fail_resp = stage_client.post(
            "/rag/ask",
            json={"query": "fastapi", "top_k": 2, "use_cache": False, "filename": "f.txt", "size": 1},
        )
        assert fail_resp.status_code == 500
    finally:
        rag_service.semantic_search_service = original_semantic
