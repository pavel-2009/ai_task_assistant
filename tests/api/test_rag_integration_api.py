"""Тесты для проверки интеграции RAG API."""

import pytest


@pytest.mark.asyncio
async def test_rag_full_cycle(authorized_client, create_all_base_tasks):
    """Тест для проверки полного цикла работы RAG API."""
    # 1. Переиндексация задач в RAG API
    response = authorized_client.post("/rag/reindex")
    assert response.status_code == 200
    assert response.json() == {"message": "Переиндексация задач запущена"}
    
    # Делаем запрос
    ask_payload = {
        "query": "Какие задачи у меня есть на сегодня по теме Python?"
    }
    
    response = authorized_client.post("/rag/ask", json=ask_payload)
    assert response.status_code == 200
    assert response.json() is not None
    assert "results" in response.json()
    
    # Проверяем, что результаты содержат ожидаемые задачи
    results = response.json()["results"]
    assert any("Задача 1" in result["title"] for result in results)
    assert any("Задача 11" in result["title"] for result in results)
    
    
@pytest.mark.asyncio
async def test_without_rag_service(authorized_client):
    """Тест для проверки обработки ошибки при отсутствии RAG-сервиса."""
    # Удаляем RAG-сервис из состояния приложения
    authorized_client.app.state.rag_service = None
    
    # Пытаемся вызвать переиндексацию
    response = authorized_client.post("/rag/reindex")
    assert response.status_code == 500
    assert response.json() == {"detail": "Ошибка при загрузке RAG-модели"}
    
    # Пытаемся вызвать запрос к RAG модели
    ask_payload = {
        "query": "Какие задачи у меня есть на сегодня по теме Python?"
    }
    
    response = authorized_client.post("/rag/ask", json=ask_payload)
    assert response.status_code == 500
    assert response.json() == {"detail": "Ошибка при загрузке RAG-модели"}