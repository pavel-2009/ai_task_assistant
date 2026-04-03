"""API-тесты CRUD-операций для задач."""

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
async def test_create_task_invalid_data(authorized_client, create_base_users):
    """Тестирование создания задачи с невалидными данными."""
    response = authorized_client.post("/tasks/", json={"title": "", "description": "This is a test task."})
    assert response.status_code == 422
    response = authorized_client.post("/tasks/", json={"title": "Test Task"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_task_success(authorized_client, create_base_users):
    """Тестирование успешного создания задачи."""
    response = authorized_client.post("/tasks/", json={"title": "Test Task", "description": "This is a test task."})
    assert response.status_code == 201
    data = response.json()
    assert data.get("title") == "Test Task"
    assert data.get("description") == "This is a test task."


@pytest.mark.asyncio
async def test_get_task_by_id_success(authorized_client, create_base_users):
    """Проверяет получение задачи по ID."""
    create_response = authorized_client.post(
        "/tasks/",
        json={"title": "By ID", "description": "task for get by id"},
    )
    assert create_response.status_code == 201
    task_id = create_response.json()["id"]

    response = authorized_client.get(f"/tasks/{task_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == task_id
    assert payload["title"] == "By ID"
    assert payload["description"] == "task for get by id"


@pytest.mark.asyncio
async def test_update_task_success(authorized_client, create_base_users):
    """Проверяет обновление задачи владельцем."""
    create_response = authorized_client.post(
        "/tasks/",
        json={"title": "Old title", "description": "Old description"},
    )
    assert create_response.status_code == 201
    task_id = create_response.json()["id"]

    response = authorized_client.put(
        f"/tasks/{task_id}",
        json={"title": "New title", "description": "New description"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == task_id
    assert payload["title"] == "New title"
    assert payload["description"] == "New description"


@pytest.mark.asyncio
async def test_update_task_forbidden_for_non_owner(
    authorized_client,
    authorized_client2,
    create_base_users,
):
    """Проверяет запрет обновления задачи чужим пользователем."""
    create_response = authorized_client.post(
        "/tasks/",
        json={"title": "Update test", "description": "Task owned by user1 for update test"},
    )
    assert create_response.status_code == 201, f"Failed to create task: {create_response.json()}"
    task_id = create_response.json().get("id")
    assert task_id is not None, f"No task_id in response: {create_response.json()}"

    response = authorized_client2.put(
        f"/tasks/{task_id}",
        json={"title": "Hacked title"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_delete_task_success(authorized_client, create_base_users):
    """Проверяет удаление задачи владельцем."""
    create_response = authorized_client.post(
        "/tasks/",
        json={"title": "Delete test", "description": "Task to be removed by owner"},
    )
    assert create_response.status_code == 201
    task_id = create_response.json()["id"]

    delete_response = authorized_client.delete(f"/tasks/{task_id}")
    assert delete_response.status_code == 204

    get_response = authorized_client.get(f"/tasks/{task_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_task_forbidden_for_non_owner(
    authorized_client,
    authorized_client2,
    create_base_users,
):
    """Проверяет запрет удаления задачи чужим пользователем."""
    create_response = authorized_client.post(
        "/tasks/",
        json={"title": "Protected task", "description": "Protected task owned by user1 for deletion test"},
    )
    assert create_response.status_code == 201, f"Failed to create task: {create_response.json()}"
    task_id = create_response.json().get("id")
    assert task_id is not None, f"No task_id in response: {create_response.json()}"

    response = authorized_client2.delete(f"/tasks/{task_id}")

    assert response.status_code == 403
