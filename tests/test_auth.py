"""
Тестирование аутентификации и авторизации в приложении FastAPI.
"""

from fastapi.testclient import TestClient
import pytest

import uuid

from app.main import app


@pytest.fixture
def client():
    """Фикстура для создания тестового клиента FastAPI"""
    with TestClient(app) as c:
        yield c


def test_user_registration_and_login(client):
    """Тестирование регистрации и входа пользователя"""

    username = f"testuser_{uuid.uuid4().hex[:8]}"
    password = "testpassword"

    # Тестируем регистрацию нового пользователя
    registration_data = {
        "username": username,
        "password": password
    }
    response = client.post("/auth/register", json=registration_data)
    assert response.status_code == 201
    assert response.json()["username"] == registration_data["username"]

    # Тестируем вход с правильными данными
    login_data = {
        "username": username,
        "password": password
    }

    response = client.post("/auth/login", data=login_data)
    assert response.status_code == 200
    assert "access_token" in response.json()

    # Тестируем вход с неправильным паролем
    wrong_login_data = {
        "username": username,
        "password": "wrongpassword"
    }
    response = client.post("/auth/login", data=wrong_login_data)
    assert response.status_code == 400
