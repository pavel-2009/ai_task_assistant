"""
Тестирование базового CV-флоу (загрузка аватара) через API.
"""

import io
import uuid

import pytest

pytest.importorskip("fastapi")
Image = pytest.importorskip("PIL.Image")
from fastapi.testclient import TestClient

from app.main import app


def _build_image_bytes() -> bytes:
    image = Image.new("RGB", (256, 256), color=(100, 150, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_upload_avatar():
    """Проверка загрузки аватара для задачи"""

    username = f"testuser_{uuid.uuid4().hex[:8]}"
    password = "testpassword"

    with TestClient(app) as client:
        registration_data = {
            "username": username,
            "password": password,
        }
        response = client.post("/auth/register", json=registration_data)
        assert response.status_code == 201

        login_data = {
            "username": username,
            "password": password,
        }
        response = client.post("/auth/login", data=login_data)
        assert response.status_code == 200

        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        task_create_response = client.post(
            "/tasks/",
            json={"title": "Test Task", "description": "Avatar upload"},
            headers=headers,
        )
        assert task_create_response.status_code == 201
        task_id = task_create_response.json()["id"]

        upload_response = client.post(
            f"/tasks/{task_id}/avatar",
            files={"image": ("test_image.jpg", _build_image_bytes(), "image/jpeg")},
            headers=headers,
        )
        assert upload_response.status_code == 200
        assert "filepath" in upload_response.json()
