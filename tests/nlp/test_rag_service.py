import asyncio
import json
from unittest.mock import AsyncMock

from app.ml.nlp.rag_service import RAGService


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttl = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttl[key] = ttl
        return True


def test_rag_service_returns_fallback_when_search_is_empty():
    llm_service = AsyncMock()
    semantic_search_service = AsyncMock()
    semantic_search_service.search.return_value = []
    service = RAGService(
        llm_service=llm_service,
        semantic_search_service=semantic_search_service,
        redis=None,
    )

    response = asyncio.run(
        service.ask("missing topic", session=AsyncMock(), top_k=3, use_cache=False)
    )

    assert response == {
        "answer": (
            "У меня нет информации об этом в ваших задачах. "
            "Попробуйте переформулировать вопрос или создать задачу по этой теме."
        ),
        "sources": [],
        "confidence": 0,
        "cached": False,
    }
    llm_service.generate.assert_not_awaited()


def test_rag_service_builds_answer_and_caches_it():
    redis = FakeRedis()
    llm_service = AsyncMock()
    llm_service.generate.return_value = "Ответ по найденным задачам"
    semantic_search_service = AsyncMock()
    semantic_search_service.search.return_value = [
        {
            "task_id": "task-1",
            "title": "API bug",
            "description": "Fix RAG route",
            "tags": ["fastapi"],
            "similarity": 0.9,
        },
        {
            "task_id": "task-2",
            "title": "Redis cache",
            "description": "Repair cache handling",
            "tags": ["redis"],
            "similarity": 0.7,
        },
    ]
    service = RAGService(
        llm_service=llm_service,
        semantic_search_service=semantic_search_service,
        redis=redis,
    )

    response = asyncio.run(
        service.ask("how to fix rag?", session=AsyncMock(), top_k=2, use_cache=True)
    )

    assert response == {
        "answer": "Ответ по найденным задачам",
        "sources": [
            {"task_id": "task-1", "title": "API bug", "similarity": 0.9},
            {"task_id": "task-2", "title": "Redis cache", "similarity": 0.7},
        ],
        "confidence": 0.8,
        "cached": False,
    }
    cache_key = service._get_cache_key("how to fix rag?", 2)
    assert json.loads(redis.store[cache_key]) == response
    assert redis.ttl[cache_key] == 300


def test_rag_service_returns_cached_response_without_llm_call():
    redis = FakeRedis()
    llm_service = AsyncMock()
    semantic_search_service = AsyncMock()
    service = RAGService(
        llm_service=llm_service,
        semantic_search_service=semantic_search_service,
        redis=redis,
    )
    cache_key = service._get_cache_key("cached question", 1)
    redis.store[cache_key] = json.dumps(
        {
            "answer": "cached answer",
            "sources": [{"task_id": "task-1", "title": "Cached", "similarity": 0.95}],
            "confidence": 0.95,
            "cached": False,
        },
        ensure_ascii=False,
    )

    response = asyncio.run(
        service.ask("cached question", session=AsyncMock(), top_k=1, use_cache=True)
    )

    assert response == {
        "answer": "cached answer",
        "sources": [{"task_id": "task-1", "title": "Cached", "similarity": 0.95}],
        "confidence": 0.95,
        "cached": True,
    }
    semantic_search_service.search.assert_not_awaited()
    llm_service.generate.assert_not_awaited()
