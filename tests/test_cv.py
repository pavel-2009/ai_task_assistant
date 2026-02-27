"""
Тестирование системы компьютерного зрения приложения FastAPI
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


def test_predict_image_class(client):
    """Тестирование предсказания класса изображения"""

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

    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with open("tests/1.jpeg", "rb") as f:
        image_bytes = f.read()

    task_create_response = client.post("/tasks", json={"title": "Test Task", "description": "dsdsds"}, headers=headers)

    assert task_create_response.status_code == 201
    task_id = task_create_response.json()["id"]

    response = client.post(f"/tasks/{task_id}/avatar", files={"image": ("test_image.jpg", image_bytes, "image/jpeg")}, headers=headers)
    assert response.status_code == 200

    response = client.post(f"/tasks/{task_id}/predict", headers=headers)
    assert response.status_code == 200
    assert "predicted_class" in response.json()
