"""API-тесты для загрузки аватаров."""

import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db_models import Task
from app.routers import avatars as avatars_router


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _DummyAvatarSession:
    def __init__(self, task):
        self.task = task

    async def execute(self, query):
        if "SELECT" in str(query).upper():
            return _ScalarResult(self.task)
        return None

    async def commit(self):
        return None


@pytest.fixture
def avatar_api_client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    app = FastAPI()
    app.include_router(avatars_router.router)

    async def _override_user():
        return SimpleNamespace(id=10)

    async def _override_session():
        yield _DummyAvatarSession(Task(id=5, title="t", description="d", author_id=10))

    app.dependency_overrides[avatars_router.get_current_user] = _override_user
    app.dependency_overrides[avatars_router.get_async_session] = _override_session

    monkeypatch.setattr(avatars_router.update_recommendations_for_task, "delay", lambda **_: None)

    return TestClient(app)


def _jpeg_bytes() -> bytes:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_upload_avatar_api_success(avatar_api_client):
    response = avatar_api_client.post(
        "/tasks/5/avatar",
        files={"image": ("avatar.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filepath"].startswith("avatars/")
    assert payload["filename"].endswith(".jpeg")


def test_upload_avatar_api_without_file(avatar_api_client):
    response = avatar_api_client.post("/tasks/5/avatar")
    assert response.status_code in (400, 422)
