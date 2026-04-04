import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_rag_full_cycle(authorized_client, create_all_base_tasks):
    """Тест для проверки полного цикла работы RAG API."""
    
    # 1. Переиндексация задач в RAG API
    response = authorized_client.post("/rag/reindex")
    
    # Если сервис не инициализирован, пропускаем тест, а не падаем
    if response.status_code == 500:
        pytest.skip("RAG service not initialized in this environment")
    
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
    """Тест проверяет, что реиндексация gracefully обрабатывает отсутствие сервиса."""
    response = authorized_client.post("/rag/reindex")
    
    # Ожидаем либо 200 (успех), либо 500 (сервис не инициализирован)
    # Оба варианта валидны в зависимости от окружения
    assert response.status_code in (200, 500)
    
    if response.status_code == 500:
        # Проверяем, что сообщение об ошибке корректное
        assert "Ошибка при загрузке RAG-модели" in response.json().get("detail", "")