from fastapi.testclient import TestClient


def test_auth_register_duplicate_user(unit_app):
    with TestClient(unit_app) as client:
        payload = {"username": "dup_router_user", "password": "Password1!"}
        first = client.post("/auth/register", json=payload)
        duplicate = client.post("/auth/register", json=payload)

    assert first.status_code == 201
    assert duplicate.status_code == 400
    assert "already exists" in duplicate.json()["detail"]


def test_nlp_validation_errors(unit_client_a: TestClient):
    too_large_top_k = unit_client_a.post("/nlp/search", json={"query": "python", "top_k": 21})
    empty_batch = unit_client_a.post("/nlp/embedding", json=[])

    assert too_large_top_k.status_code == 400
    assert "top_k" in too_large_top_k.json()["detail"]
    assert empty_batch.status_code == 400
    assert "Список текстов не может быть пустым" in empty_batch.json()["detail"]


def test_task_router_list_get_delete_cycle(unit_client_a: TestClient):
    created = unit_client_a.post(
        "/tasks/",
        json={"title": "TaskRoute", "description": "router flow"},
    )
    task_id = created.json()["id"]

    listed = unit_client_a.get("/tasks/")
    fetched = unit_client_a.get(f"/tasks/{task_id}")
    deleted = unit_client_a.delete(f"/tasks/{task_id}")
    status_after_delete = unit_client_a.get(f"/tasks/{task_id}/tags_status")

    assert listed.status_code == 200
    assert any(task["id"] == task_id for task in listed.json())
    assert fetched.status_code == 200
    assert deleted.status_code == 204
    assert status_after_delete.status_code == 404


def test_rag_stream_returns_sse_payload(unit_client_a: TestClient):
    response = unit_client_a.post(
        "/rag/ask/stream",
        json={"query": "stream this", "top_k": 2, "use_cache": False},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data: hello" in response.text
    assert "event: done" in response.text
