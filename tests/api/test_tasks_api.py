"""API-тесты для эндпоинтов задач."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routers import tasks as tasks_router


class _DummySession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        obj.id = 1
        self.added.append(obj)

    async def commit(self):
        return None

    async def refresh(self, _obj):
        return None


@pytest.fixture
def task_api_client(monkeypatch):
    app = FastAPI()
    app.include_router(tasks_router.router)

    async def _override_user():
        return SimpleNamespace(id=77)

    async def _override_session():
        yield _DummySession()

    app.dependency_overrides[tasks_router.get_current_user] = _override_user
    app.dependency_overrides[tasks_router.get_async_session] = _override_session

    monkeypatch.setattr(tasks_router.process_task_tags_and_embedding, "delay", lambda **_: None)
    monkeypatch.setattr(tasks_router.process_task_interaction, "delay", lambda **_: None)
    monkeypatch.setattr(tasks_router.update_recommendations_for_task, "delay", lambda **_: None)

    return TestClient(app)


def test_create_task_api_success(task_api_client):
    response = task_api_client.post("/tasks/", json={"title": "Task", "description": "Desc"})
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == 1
    assert data["author_id"] == 77
    assert data["title"] == "Task"


def test_create_task_api_validation_error(task_api_client):
    response = task_api_client.post("/tasks/", json={"title": "", "description": "Desc"})
    assert response.status_code == 422
