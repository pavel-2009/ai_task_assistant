"""API-тесты для модуля auth."""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone

import jwt

from app.core import config


# Test token expiration first (before rate limiting tests exhaust the limit)
@pytest.mark.asyncio
async def test_0_login_token_expiration(fresh_app_client):
    """Тестирование истечения срока действия токена."""
    # Register user
    reg_response = fresh_app_client.post("/auth/register", json={"username": "tokenuser", "password": "TokenTest123!"})
    if reg_response.status_code != 201:
        # User might already exist from previous run, try to login anyway
        pass
    
    response = fresh_app_client.post("/auth/login", data={"username": "tokenuser", "password": "TokenTest123!"})
    assert response.status_code == 200, f"Login failed: {response.json()}"
    token = response.json().get("access_token")
    assert token is not None
    
    await asyncio.sleep(65)  # Подождать, пока токен истечет (JWT_EXPIRE_MINUTES = 1)
    
    response = fresh_app_client.post("/tasks/", data={}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login(client, create_base_users):
    """Тестирование успешного входа."""
    response = client.post("/auth/login", data={"username": "testuser", "password": "TestPass123!"})
    assert response.status_code == 200
    assert "access_token" in response.json()


@pytest.mark.asyncio
async def test_auth_full_cycle_with_jwt(fresh_app_client):
    """Полный цикл: регистрация -> логин -> доступ к защищенному эндпоинту по JWT."""
    register_response = fresh_app_client.post(
        "/auth/register",
        json={"username": "fullcycleuser", "password": "FullCycle123!"},
    )
    assert register_response.status_code == 201

    login_response = fresh_app_client.post(
        "/auth/login",
        data={"username": "fullcycleuser", "password": "FullCycle123!"},
    )
    assert login_response.status_code == 200
    token = login_response.json().get("access_token")
    assert token is not None

    task_response = fresh_app_client.post(
        "/tasks/",
        json={"title": "JWT smoke", "description": "Protected call"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert task_response.status_code == 201
    task_payload = task_response.json()
    assert task_payload["title"] == "JWT smoke"
    assert task_payload["author_id"] == register_response.json()["id"]


@pytest.mark.asyncio
async def test_jwt_rejects_tampered_token(fresh_app_client):
    """Проверка, что API отклоняет JWT c поврежденной подписью."""
    fresh_app_client.post(
        "/auth/register",
        json={"username": "testuser", "password": "TestPass123!"},
    )
    login_response = fresh_app_client.post(
        "/auth/login",
        data={"username": "testuser", "password": "TestPass123!"},
    )
    assert login_response.status_code == 200
    original_token = login_response.json()["access_token"]
    tampered_token = f"{original_token}tampered"

    response = fresh_app_client.post(
        "/tasks/",
        json={"title": "bad token", "description": "must fail"},
        headers={"Authorization": f"Bearer {tampered_token}"},
    )
    assert response.status_code == 401
    assert "signature" in response.json().get("detail", "").lower()


@pytest.mark.asyncio
async def test_jwt_rejects_expired_token_without_sleep(fresh_app_client):
    """Проверка, что API отклоняет просроченный JWT без ожидания в тесте."""
    register_response = fresh_app_client.post(
        "/auth/register",
        json={"username": "expireduser", "password": "Expired123!"},
    )
    assert register_response.status_code == 201
    expired_payload = {
        "user_id": register_response.json()["id"],
        "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
    }
    expired_token = jwt.encode(expired_payload, config.SECRET_KEY, algorithm=config.JWT_ALGORITHM)

    response = fresh_app_client.post(
        "/tasks/",
        json={"title": "expired token", "description": "must fail"},
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401
    assert "expired" in response.json().get("detail", "").lower()

    
@pytest.mark.asyncio
async def test_login_invalid_credentials(client, create_base_users):
    """Тестирование входа с неверными учетными данными."""
    response = client.post("/auth/login", data={"username": "testuser", "password": "wrongpass"})
    assert response.status_code == 400
    assert response.json().get("detail") == "Неверное имя пользователя или пароль"
    
    
@pytest.mark.asyncio
async def test_login_nonexistent_user(client):
    """Тестирование входа с несуществующим пользователем."""
    response = client.post("/auth/login", data={"username": "nonexistent", "password": "testpass"})
    assert response.status_code == 400
    assert response.json().get("detail") == "Неверное имя пользователя или пароль"
    
    
@pytest.mark.asyncio
async def test_login_missing_fields(client):
    """Тестирование входа с отсутствующими полями."""
    response = client.post("/auth/login", data={"username": "testuser"})
    assert response.status_code == 422  # Unprocessable Entity
    response = client.post("/auth/login", data={"password": "testpass"})
    assert response.status_code == 422  # Unprocessable Entity
    
    
@pytest.mark.asyncio
async def test_login_empty_fields(client):
    """Тестирование входа с пустыми полями."""
    response = client.post("/auth/login", data={"username": "", "password": "testpass"})
    assert response.status_code == 422  # Unprocessable Entity
    assert response.json().get("detail") == "Invalid request parameters"
    response = client.post("/auth/login", data={"username": "testuser", "password": ""})
    assert response.status_code == 422  # Unprocessable Entity
    assert response.json().get("detail") == "Invalid request parameters"
    
    
@pytest.mark.asyncio
async def test_login_sql_injection(client):
    """Тестирование входа с попыткой SQL-инъекции."""
    response = client.post("/auth/login", data={"username": "testuser' OR '1'='1", "password": "testpass"})
    assert response.status_code == 400
    assert response.json().get("detail") == "Неверное имя пользователя или пароль"
    
    
@pytest.mark.asyncio
async def test_login_brute_force(client2, create_base_users):
    """Тестирование защиты от перебора паролей."""
    for _ in range(5):
        response = client2.post("/auth/login", data={"username": "testuser", "password": "wrongpass"})
        assert response.status_code == 400
        assert response.json().get("detail") == "Неверное имя пользователя или пароль"
        
        
@pytest.mark.asyncio
async def test_rate_limiting(fresh_app_client, create_base_users):
    """Тестирование ограничения количества запросов."""
    # Register user для fresh_app_client
    fresh_app_client.post("/auth/register", json={"username": "testuser", "password": "TestPass123!"})
    for _ in range(12):
        response = fresh_app_client.post("/auth/login", data={"username": "testuser", "password": "wrongpass"})
    assert response.status_code == 429  # Too Many Requests
    assert "Rate limit exceeded" in response.json().get("error", "")
    
@pytest.mark.asyncio
async def test_login_unexisting_user(client):
    """Тестирование входа с несуществующим пользователем."""
    response = client.post("/auth/login", data={"username": "ghostuser", "password": "GhostPass123!"})
    assert response.status_code == 400
    assert response.json().get("detail") == "Неверное имя пользователя или пароль"

    
# Тестирование регистрации
@pytest.mark.asyncio
async def test_registration(client):
    """Тестирование успешной регистрации."""
    response = client.post("/auth/register", json={"username": "newuser", "password": "NewPass123!"})
    assert response.status_code == 201
    assert response.json().get("username") == "newuser"
    assert response.json().get("id") is not None
    
    
@pytest.mark.asyncio
async def test_registration_existing_user(client, create_base_users):
    """Тестирование регистрации с уже существующим именем пользователя."""
    response = client.post("/auth/register", json={"username": "testuser", "password": "TestPass123!"})
    assert response.status_code == 400
    assert response.json().get("detail") == "Пользователь с таким именем уже существует"
    

@pytest.mark.asyncio
async def test_registration_invalid_password(client):
    """Тестирование регистрации с недопустимым паролем."""
    response = client.post("/auth/register", json={"username": "user2", "password": "short"})
    assert response.status_code == 422  # Pydantic validation error
    
    
@pytest.mark.asyncio
async def test_registration_missing_fields(client):
    """Тестирование регистрации с отсутствующими полями."""
    response = client.post("/auth/register", json={"username": "user3"})
    assert response.status_code == 422  # Unprocessable Entity
    response = client.post("/auth/register", json={"password": "TestPass123!"})
    assert response.status_code == 422  # Unprocessable Entity
        
