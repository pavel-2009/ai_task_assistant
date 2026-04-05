from fastapi.testclient import TestClient


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post(
        "/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_register_login_and_create_task(unit_client_a: TestClient):
    token = _token(unit_client_a, "user_a", "Password1!")
    response = unit_client_a.post(
        "/tasks/",
        json={"title": "Task1", "description": "Build feature in FastAPI with Redis"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Task1"
    assert data["author_id"] > 0


def test_task_owner_cannot_be_changed_by_another_user(unit_client_a: TestClient, unit_client_b: TestClient):
    token_a = _token(unit_client_a, "user_a", "Password1!")
    created = unit_client_a.post(
        "/tasks/",
        json={"title": "TaskOwn", "description": "Owner only"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    task_id = created.json()["id"]

    token_b = _token(unit_client_b, "user_b", "Password1!")
    forbidden = unit_client_b.put(
        f"/tasks/{task_id}",
        json={"title": "Hacked"},
        headers={"Authorization": f"Bearer {token_b}"},
    )

    assert forbidden.status_code == 403


def test_like_and_status_endpoints(unit_client_a: TestClient):
    token = _token(unit_client_a, "user_a", "Password1!")
    created = unit_client_a.post(
        "/tasks/",
        json={"title": "TaskLike", "description": "FastAPI Celery"},
        headers={"Authorization": f"Bearer {token}"},
    )
    task_id = created.json()["id"]

    like_resp = unit_client_a.post(f"/tasks/{task_id}/like", headers={"Authorization": f"Bearer {token}"})
    status_resp = unit_client_a.get(f"/tasks/{task_id}/tags_status")

    assert like_resp.status_code == 200
    assert like_resp.json()["message"]
    assert status_resp.status_code == 200
    assert status_resp.json()["is_processing"] is True
