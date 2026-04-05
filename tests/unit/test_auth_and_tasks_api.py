from fastapi.testclient import TestClient


def test_register_login_and_create_task(unit_client_a: TestClient):
    response = unit_client_a.post(
        "/tasks/",
        json={"title": "Task1", "description": "Build feature in FastAPI with Redis"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Task1"
    assert data["author_id"] > 0


def test_task_owner_cannot_be_changed_by_another_user(unit_client_a: TestClient, unit_client_b: TestClient):
    created = unit_client_a.post(
        "/tasks/",
        json={"title": "TaskOwn", "description": "Owner only"},
    )
    task_id = created.json()["id"]

    forbidden = unit_client_b.put(
        f"/tasks/{task_id}",
        json={"title": "Hacked"},
    )

    assert forbidden.status_code == 403


def test_like_and_status_endpoints(unit_client_a: TestClient):
    created = unit_client_a.post(
        "/tasks/",
        json={"title": "TaskLike", "description": "FastAPI Celery"},
    )
    task_id = created.json()["id"]

    like_resp = unit_client_a.post(f"/tasks/{task_id}/like")
    status_resp = unit_client_a.get(f"/tasks/{task_id}/tags_status")

    assert like_resp.status_code == 200
    assert like_resp.json()["message"]
    assert status_resp.status_code == 200
    assert status_resp.json()["is_processing"] is True
