import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_rag_full_cycle(authorized_client, create_all_base_tasks):
    """Тест для проверки полного цикла работы RAG API."""
    
    # 1. Переиндексация задач в RAG API
    response = authorized_client.post("/rag/reindex")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Переиндексация задач запущена"}
    
    # Даем время на индексацию
    import time
    time.sleep(1)
    
    ask_payload = {
        "query": "Какие задачи у меня есть на сегодня по теме Python?"
    }
    
    response = authorized_client.post("/rag/ask", json=ask_payload)
    assert response.status_code == 200
    assert response.json() is not None


@pytest.mark.asyncio
async def test_rag_reindex_graceful_fallback(authorized_client, create_all_base_tasks):
    """Тест проверяет, что эндпоинт реиндексации успешно запускается."""
    response = authorized_client.post("/rag/reindex")
    assert response.status_code == 200
    assert response.json() == {"message": "Переиндексация задач запущена"}
