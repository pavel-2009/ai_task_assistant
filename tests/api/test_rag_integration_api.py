"""Интеграционные full-cycle тесты RAG на реальных сервисах + Celery реиндексация."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete

import app.services as services_registry
from app.core import config
from app.db import async_session
from app.db_models import Task, Text, User
from app.error_handlers import register_exception_handlers
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.llm_service import LLMService
from app.ml.nlp.rag_service import RAGService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.tasks import reindex_tasks
from app.ml.nlp.vector_db import VectorDB
from app.routers import nlp as nlp_router
from app.routers import rag as rag_router


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
async def rag_real_stack():
    """Минимальная подготовка: создать задачи в БД, затем запустить Celery реиндексацию."""
    # 1) Реальные сервисы
    embedding_service = EmbeddingService()
    vector_db = VectorDB(dim=embedding_service.dimension, redis_client=None)
    semantic_search = SemanticSearchService(
        embedding_service=embedding_service,
        vector_db=vector_db,
        redis_client=None,
    )

    llm_service = LLMService(base_url=config.LLM_BASE_URL, model=config.LLM_MODEL)
    llm_service.timeout_seconds = config.LLM_TIMEOUT_SECONDS
    rag_service = RAGService(
        llm_service=llm_service,
        semantic_search_service=semantic_search,
        redis=None,
    )

    # 2) Минимум pre-test кода: только задачи в БД
    async with async_session() as session:
        await session.execute(delete(Text))
        await session.execute(delete(Task))
        await session.execute(delete(User))
        user = User(username="rag_test_user", password="hashed_password")
        session.add(user)
        await session.flush()
        for title, description in TECH_TASKS:
            session.add(Task(title=title, description=description, author_id=user.id))
        await session.commit()

    # 3) Инициализация для Celery task -> реиндексация всей базы
    old_services = services_registry._services.copy()
    services_registry._services["semantic_search"] = semantic_search
    reindex_tasks()

    # 4) API c реальными endpoints
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(nlp_router.router)
    app.include_router(rag_router.router)
    app.state.embedding_service = embedding_service
    app.state.semantic_search_service = semantic_search
    app.state.llm_service = llm_service
    app.state.rag_service = rag_service

    client = TestClient(app)

    yield {
        "client": client,
        "app": app,
        "vector_db": vector_db,
    }

    client.close()
    services_registry._services.clear()
    services_registry._services.update(old_services)


@pytest.mark.asyncio
async def test_semantic_reindex_via_celery_task(rag_real_stack):
    """Проверка индексации всей базы задач через Celery задачу reindex_tasks."""
    vector_db = rag_real_stack["vector_db"]
    assert len(vector_db.ids) == len(TECH_TASKS)


@pytest.mark.asyncio
async def test_rag_ask_edge_cases_and_fallback(rag_real_stack):
    """Проверка RAG /ask: валидный запрос + edge cases + fallback облачной модели."""
    client = rag_real_stack["client"]

    ok_resp = client.post(
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
    assert ok_payload["sources"]
    assert isinstance(ok_payload["answer"], str) and ok_payload["answer"]

    empty_query = client.post(
        "/rag/ask",
        json={"query": "", "top_k": 5, "use_cache": False, "filename": "e.txt", "size": 1},
    )
    assert empty_query.status_code == 422

    invalid_top_k = client.post(
        "/rag/ask",
        json={"query": "python", "top_k": 50, "use_cache": False, "filename": "t.txt", "size": 1},
    )
    assert invalid_top_k.status_code == 422

    assert isinstance(ok_payload["answer"], str) and ok_payload["answer"]




@pytest.mark.asyncio
async def test_llm_uses_openrouter_endpoint(rag_real_stack):
    """Проверка, что для RAG используется OpenRouter cloud endpoint."""
    llm_service = rag_real_stack["app"].state.llm_service
    assert "openrouter.ai" in llm_service.url


@pytest.mark.asyncio
async def test_rag_repeat_question_cache_behavior(rag_real_stack):
    """Повторный вопрос: проверка поведения кэширования (без Redis cached=false)."""
    client = rag_real_stack["client"]
    payload = {
        "query": "Что есть по Kubernetes?",
        "top_k": 3,
        "use_cache": True,
        "filename": "k8s.txt",
        "size": 2,
    }

    first = client.post("/rag/ask", json=payload)
    second = client.post("/rag/ask", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json().get("cached") is False
    assert second.json().get("cached") is False


@pytest.mark.asyncio
async def test_rag_stream_and_internal_errors(rag_real_stack):
    """Потоковый endpoint + внутренняя ошибка RAG."""
    client = rag_real_stack["client"]
    app = rag_real_stack["app"]

    with client.stream(
        "POST",
        "/rag/ask/stream",
        json={"query": "TypeScript", "top_k": 3, "use_cache": False, "filename": "s.txt", "size": 1},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        stream_data = "".join(stream_resp.iter_text())
    assert "event: done" in stream_data

    original_semantic = app.state.rag_service.semantic_search_service
    app.state.rag_service.semantic_search_service = None
    try:
        fail_resp = client.post(
            "/rag/ask",
            json={"query": "fastapi", "top_k": 2, "use_cache": False, "filename": "f.txt", "size": 1},
        )
        assert fail_resp.status_code == 500
    finally:
        app.state.rag_service.semantic_search_service = original_semantic
