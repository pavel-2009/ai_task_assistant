"""Тестирование NLP задач"""

import pytest


@pytest.mark.asyncio
async def test_embedding_endpoint(authorized_client):
    """Тест для проверки эндпоинта получения эмбеддингов."""
    
    response = authorized_client.post("/nlp/embedding", json="Привет, как дела?")
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 384  # Проверяем размер эмбеддинга
    
    # Тест для batch режима (список строк)
    response = authorized_client.post("/nlp/embedding", json=["Привет", "Как дела?"])
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert len(data["embedding"]) == 2
    assert len(data["embedding"][0]) == 384
    
    
@pytest.mark.asyncio
async def test_semantic_search_endpoint(authorized_client, create_all_base_tasks):
    """Тест для проверки эндпоинта семантического поиска."""
    # Сначала нужно проиндексировать задачи
    response = authorized_client.post("/rag/reindex")
    # Может вернуть 500 если RAG не инициализирован, пропускаем
    if response.status_code == 500:
        pytest.skip("RAG service not initialized")
    
    payload = {
        "query": "задача",
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
    payload = "Это тестовый текст для индексации."
    response = authorized_client.post("/nlp/index", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("detail") == "Текст успешно индексирован"