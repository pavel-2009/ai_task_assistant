"""Простые API-тесты загрузки аватаров."""

import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import insert

from app.db_models import Task
from app.routers import avatars as avatars_router


async def _seed_task(sessionmaker, task_id: int, author_id: int):
    async with sessionmaker() as session:
        await session.execute(
            insert(Task).values(id=task_id, title="t", description="d", author_id=author_id)
        )
        await session.commit()


def _jpeg_bytes() -> bytes:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


@pytest.fixture
def avatars_client(api_sessionmaker, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Ключевой комментарий: используем реальный test.db, а не самописные фейк-сессии.
    import asyncio
    asyncio.run(_seed_task(api_sessionmaker, task_id=5, author_id=10))

    app = FastAPI()
    app.include_router(avatars_router.router)

    async def _override_user():
        return SimpleNamespace(id=10)

    async def _override_session():
        async with api_sessionmaker() as session:
            yield session

    app.dependency_overrides[avatars_router.get_current_user] = _override_user
    app.dependency_overrides[avatars_router.get_async_session] = _override_session
    monkeypatch.setattr(avatars_router.update_recommendations_for_task, "delay", lambda **_: None)

    return TestClient(app)


def test_upload_avatar_success(avatars_client):
    response = avatars_client.post(
        "/tasks/5/avatar",
        files={"image": ("avatar.jpg", io.BytesIO(_jpeg_bytes()), "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filepath"].startswith("avatars/")


def test_upload_avatar_without_file(avatars_client):
    response = avatars_client.post("/tasks/5/avatar")
    assert response.status_code in (400, 422)
