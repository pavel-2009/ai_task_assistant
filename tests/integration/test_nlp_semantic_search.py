import pytest


@pytest.mark.integration
def test_nlp_embedding_index_search_and_tag(integration_httpx_client_a, seeded_tasks_and_index):
    embedding = integration_httpx_client_a.post("/nlp/embedding", json="FastAPI + Redis semantic query")
    assert embedding.status_code == 200, embedding.text

    index_text = "Semantic search over Python FastAPI Celery and Redis stack"
    index_resp = integration_httpx_client_a.post("/nlp/index", json={"text": index_text})
    assert index_resp.status_code == 200, index_resp.text

    search_resp = integration_httpx_client_a.post(
        "/nlp/search",
        json={"query": "FastAPI Redis", "top_k": 5},
    )
    assert search_resp.status_code == 200, search_resp.text
    assert search_resp.json()["total"] >= 1

    tag_resp = integration_httpx_client_a.post(
        "/nlp/tag-task",
        json="Build NLP service with Python, FastAPI, Redis and Celery",
    )
    assert tag_resp.status_code == 200, tag_resp.text
    assert isinstance(tag_resp.json().get("tags"), list)


@pytest.mark.integration
def test_nlp_edge_cases(integration_httpx_client_a):
    empty_embedding = integration_httpx_client_a.post("/nlp/embedding", json="")
    assert empty_embedding.status_code == 400

    too_large_topk = integration_httpx_client_a.post(
        "/nlp/search",
        json={"query": "text", "top_k": 999},
    )
    assert too_large_topk.status_code == 400

    invalid_index = integration_httpx_client_a.post("/nlp/index", json={"text": "   "})
    assert invalid_index.status_code == 400
