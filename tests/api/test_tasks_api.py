"""Простые API-тесты создания задач."""
import pytest


@pytest.mark.asyncio
async def test_get_tasks(client, create_base_users):
    """Тестирование получения списка задач."""
    response = client.get("/tasks/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    
    
@pytest.mark.asyncio
async def test_create_task_unauthorized(client, create_base_users):
    """Тестирование создания задачи без авторизации."""
    response = client.post("/tasks/", json={"title": "Test Task", "description": "This is a test task."})
    assert response.status_code == 401
    
    
@pytest.mark.asyncio
async def test_create_task_invalid_data(authorized_client, create_base_users, auth_token):
    """Тестирование создания задачи с невалидными данными."""
    response = authorized_client.post("/tasks/", json={"title": "", "description": "This is a test task."})
    assert response.status_code == 422  # Unprocessable Entity
    response = authorized_client.post("/tasks/", json={"title": "Test Task"})
    assert response.status_code == 422  # Unprocessable Entity
    
    
@pytest.mark.asyncio
async def test_create_task_success(authorized_client, create_base_users, auth_token):
    """Тестирование успешного создания задачи."""
    response = authorized_client.post("/tasks/", json={"title": "Test Task", "description": "This is a test task."})
    assert response.status_code == 201
    data = response.json()
    assert data.get("title") == "Test Task"
    assert data.get("description") == "This is a test task."
    
    

