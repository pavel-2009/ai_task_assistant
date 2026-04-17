"""Простые API-тесты создания задач."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import tasks as tasks_router


@pytest.fixture
def tasks_client(api_sessionmaker, monkeypatch):
    app = FastAPI()
    app.include_router(tasks_router.router)

    # Ключевой комментарий: для API теста подменяем только auth и фоновые delay.
    async def _override_user():
        return SimpleNamespace(id=1)

    async def _override_session():
        async with api_sessionmaker() as session:
            yield session

    app.dependency_overrides[tasks_router.get_current_user] = _override_user
    app.dependency_overrides[tasks_router.get_async_session] = _override_session

    monkeypatch.setattr(tasks_router.process_task_tags_and_embedding, "delay", lambda **_: None)
    monkeypatch.setattr(tasks_router.process_task_interaction, "delay", lambda **_: None)
    monkeypatch.setattr(tasks_router.update_recommendations_for_task, "delay", lambda **_: None)

    return TestClient(app)


def test_create_task_success(tasks_client):
    response = tasks_client.post("/tasks/", json={"title": "Task", "description": "Desc"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["title"] == "Task"
    assert payload["author_id"] == 1


def test_create_task_invalid_title(tasks_client):
    response = tasks_client.post("/tasks/", json={"title": "", "description": "Desc"})

    assert response.status_code == 422
