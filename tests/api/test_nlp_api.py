"""Простые API-тесты NLP-роутера без RAG."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import nlp as nlp_router


class FakeEmbeddingService:
    def encode_one(self, _text):
        return np.array([0.1, 0.2], dtype=np.float32)

    def encode_batch(self, texts):
        return np.array([[0.1, 0.2] for _ in texts], dtype=np.float32)


class FakeSemanticSearchService:
    async def search(self, _query, session, top_k=5):
        return [{"text_id": "1", "similarity": 0.9, "text": "hello"}][:top_k]

    async def index(self, _text, session):
        return "1"


class FakeNerService:
    def tag_task(self, _text):
        return {"technologies": [("python", 0.9), ("fastapi", 0.95)]}


@pytest.fixture
def nlp_client(api_sessionmaker):
    app = FastAPI()
    app.include_router(nlp_router.router)

    app.state.embedding_service = FakeEmbeddingService()
    app.state.semantic_search_service = FakeSemanticSearchService()
    app.state.ner_service = FakeNerService()

    # Ключевой комментарий: для зависимостей БД используем session из test.db.
    async def _override_session():
        async with api_sessionmaker() as session:
            yield session

    app.dependency_overrides[nlp_router.get_async_session] = _override_session

    return TestClient(app)


def test_embedding_single_text(nlp_client):
    response = nlp_client.post("/nlp/embedding", json="some text")
    assert response.status_code == 200
    assert response.json()["embedding"] == [0.1, 0.2]


def test_search_returns_results(nlp_client):
    response = nlp_client.post("/nlp/search", json={"query": "hello", "top_k": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["results"][0]["text_id"] == "1"


def test_tag_task_returns_tags(nlp_client):
    response = nlp_client.post("/nlp/tag-task", json="Build API with FastAPI")
    assert response.status_code == 200
    assert response.json()["tags"] == ["python", "fastapi"]
