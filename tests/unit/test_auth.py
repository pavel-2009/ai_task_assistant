"""Юнит-тесты для модуля auth."""

import pytest
import asyncio

from tests import conftest


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
    assert response.status_code == 422
    assert response.json().get("detail") == "Invalid request parameters"
    response = client.post("/auth/login", data={"username": "testuser", "password": ""})
    assert response.status_code == 422
    assert response.json().get("detail") == "Invalid request parameters"
    
    
@pytest.mark.asyncio
async def test_login_sql_injection(client):
    """Тестирование входа с попыткой SQL-инъекции."""
    response = client.post("/auth/login", data={"username": "testuser' OR '1'='1", "password": "testpass"})
    assert response.status_code == 400
    assert response.json().get("detail") == "Неверное имя пользователя или пароль"
    
    
@pytest.mark.asyncio
async def test_login_brute_force(client, create_base_users):
    """Тестирование защиты от перебора паролей."""
    for _ in range(5):
        response = client.post("/auth/login", data={"username": "testuser", "password": "wrongpass"})
        assert response.status_code == 400
        assert response.json().get("detail") == "Неверное имя пользователя или пароль"
        
        
@pytest.mark.asyncio
async def test_rate_limiting(client, create_base_users):
    """Тестирование ограничения количества запросов."""
    for _ in range(12):
        response = client.post("/auth/login", data={"username": "testuser", "password": "wrongpass"})
    assert response.status_code == 429  # Too Many Requests
    assert "Rate limit exceeded" in response.json().get("error", "")