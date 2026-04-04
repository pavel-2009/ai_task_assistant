"""Тестирование NLP задач"""

import pytest


@pytest.mark.asyncio
async def test_embedding_endpoint(authorized_client):
    """Тест для проверки эндпоинта получения эмбеддингов."""
    payload = {
        "texts": [
            "Привет, как дела?",
            "Какая погода сегодня?"
        ]
    }
    response = authorized_client.post("/nlp/embedding", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    assert all(len(embedding) == 384 for embedding in data["embeddings"])
    
    
@pytest.mark.asyncio
async def test_semantic_search_endpoint(authorized_client):
    """Тест для проверки эндпоинта семантического поиска."""
    payload = {
        "query": "Какой сегодня день?",
        "top_k": 3
    }
    response = authorized_client.post("/nlp/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= 3
    
    
@pytest.mark.asyncio
async def test_index_endpoint(authorized_client):
    """Тест для проверки эндпоинта индексации текста."""
    payload = {
        "text": "Это тестовый текст для индексации."
    }
    response = authorized_client.post("/nlp/index", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert len(data["embedding"]) == 384
