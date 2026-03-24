"""
Тестирование аутентификации и авторизации в приложении FastAPI.
"""

import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.main import app


def test_user_registration_and_login():
    """Тестирование регистрации и входа пользователя"""

    username = f"testuser_{uuid.uuid4().hex[:8]}"
    password = "testpassword"

    with TestClient(app) as client:
        registration_data = {
            "username": username,
            "password": password,
        }
        response = client.post("/auth/register", json=registration_data)
        assert response.status_code == 201
        assert response.json()["username"] == registration_data["username"]

        login_data = {
            "username": username,
            "password": password,
        }
        response = client.post("/auth/login", data=login_data)
        assert response.status_code == 200
        assert "access_token" in response.json()

        wrong_login_data = {
            "username": username,
            "password": "wrongpassword",
        }
        response = client.post("/auth/login", data=wrong_login_data)
        assert response.status_code == 400
