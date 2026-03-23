"""
Базовые тесты на CRUD задач.
"""

import uuid

import pytest


class StubNerService:
    def tag_task(self, text: str) -> dict:
        return {
            "technologies": [("stub-tech", 0.9)] if text else [],
            "confidence": 0.9 if text else 0,
        }


def _auth_headers(client) -> dict[str, str]:
    username = f"task_user_{uuid.uuid4().hex[:8]}"
    password = "testpassword"

    register_payload = {
        "username": username,
        "password": password,
    }
    register_response = client.post("/auth/register", json=register_payload)
    assert register_response.status_code == 201

    login_payload = {
        "username": username,
        "password": password,
    }
    login_response = client.post("/auth/login", data=login_payload)
    assert login_response.status_code == 200

    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_task_crud_flow():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        client.app.state.ner_service = StubNerService()
        headers = _auth_headers(client)

        create_payload = {
            "title": "Test Task",
            "description": "Test description",
        }
        create_response = client.post("/tasks/", json=create_payload, headers=headers)
        assert create_response.status_code == 201

        created_task = create_response.json()
        task_id = created_task["id"]
        assert created_task["title"] == create_payload["title"]

        get_response = client.get(f"/tasks/{task_id}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == task_id

        list_response = client.get("/tasks/")
        assert list_response.status_code == 200
        assert any(task["id"] == task_id for task in list_response.json())

        update_payload = {
            "title": "Updated Task",
            "description": "Updated description",
        }
        update_response = client.put(f"/tasks/{task_id}", json=update_payload, headers=headers)
        assert update_response.status_code == 200
        assert update_response.json()["title"] == update_payload["title"]

        delete_response = client.delete(f"/tasks/{task_id}", headers=headers)
        assert delete_response.status_code == 204

        get_deleted_response = client.get(f"/tasks/{task_id}")
        assert get_deleted_response.status_code == 404
