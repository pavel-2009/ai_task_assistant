from fastapi.testclient import TestClient


def test_embedding_single_and_batch(unit_client_a: TestClient):
    single = unit_client_a.post("/nlp/embedding", json="FastAPI pipeline")
    batch = unit_client_a.post("/nlp/embedding", json=["one", "two"])

    assert single.status_code == 200
    assert len(single.json()["embedding"]) == 4
    assert batch.status_code == 200
    assert len(batch.json()["embedding"]) == 2


def test_nlp_index_search_and_tag(unit_client_a: TestClient):
    index_resp = unit_client_a.post("/nlp/index", json={"text": "Task about Python FastAPI Redis"})
    search_resp = unit_client_a.post("/nlp/search", json={"query": "Python", "top_k": 5})
    tag_resp = unit_client_a.post("/nlp/tag-task", json="Build API with Python FastAPI and Redis")

    assert index_resp.status_code == 200
    assert search_resp.status_code == 200
    assert search_resp.json()["total"] >= 1
    assert tag_resp.status_code == 200
    assert "python" in tag_resp.json()["tags"]


def test_rag_ask_and_reindex(unit_client_a: TestClient):
    ask_resp = unit_client_a.post("/rag/ask", json={"query": "What is in my tasks?", "top_k": 3, "use_cache": True})
    reindex_resp = unit_client_a.post("/rag/reindex")

    assert ask_resp.status_code == 200
    assert ask_resp.json()["answer"]
    assert reindex_resp.status_code == 200
