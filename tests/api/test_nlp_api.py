"""API-тесты для NLP роутера (без RAG)."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import nlp as nlp_router


class _EmbeddingService:
    def encode_one(self, text):
        return SimpleNamespace(tolist=lambda: [0.1, 0.2], __iter__=lambda self: iter([0.1, 0.2]))

    def encode_batch(self, texts):
        return SimpleNamespace(tolist=lambda: [[0.1, 0.2] for _ in texts])


class _SemanticSearchService:
    async def search(self, query, session, top_k=5):
        return [{"text_id": "1", "similarity": 0.9, "text": "hello"}][:top_k]

    async def index(self, text, session):
        return "1"


class _NerService:
    def tag_task(self, _text):
        return {"technologies": [("python", 0.9), ("fastapi", 0.95)]}


@pytest.fixture
def nlp_api_client():
    app = FastAPI()
    app.include_router(nlp_router.router)

    app.state.embedding_service = _EmbeddingService()
    app.state.semantic_search_service = _SemanticSearchService()
    app.state.ner_service = _NerService()

    async def _override_session():
        yield object()

    app.dependency_overrides[nlp_router.get_async_session] = _override_session

    return TestClient(app)


def test_embedding_api_single(nlp_api_client):
    response = nlp_api_client.post("/nlp/embedding", json="some text")
    assert response.status_code == 200
    assert response.json()["embedding"] == [0.1, 0.2]


def test_search_api_success(nlp_api_client):
    response = nlp_api_client.post("/nlp/search", json={"query": "hello", "top_k": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["results"][0]["text_id"] == "1"


def test_tag_task_api_success(nlp_api_client):
    response = nlp_api_client.post("/nlp/tag-task", json="Build API with FastAPI")
    assert response.status_code == 200
    assert response.json()["tags"] == ["python", "fastapi"]
