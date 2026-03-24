"""
Тесты для векторной БД на FAISS.
"""


import json
from unittest.mock import AsyncMock

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")
pytest.importorskip("redis")
pytest.importorskip("sqlalchemy")

from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.vector_db import VectorDB


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttl = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value):
        self.store[key] = value
        return True

    async def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttl[key] = ttl
        return True

    async def scan(self, cursor, match=None):
        prefix = match[:-1] if match and match.endswith("*") else match
        keys = [key for key in self.store if prefix is None or key.startswith(prefix)]
        return 0, keys

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
            self.ttl.pop(key, None)
        return len(keys)

    def pipeline(self, transaction=True):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def set(self, key, value):
        self.commands.append((key, value))
        return self

    async def execute(self):
        for key, value in self.commands:
            self.redis.store[key] = value
        return True


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class StubEmbeddingService:
    dimension = 3

    def encode_one(self, text: str) -> np.ndarray:
        base = {
            "doc-1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "doc-2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "query": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
        return base[text]


@pytest.mark.asyncio
async def test_vector_db_persists_ids_and_returns_text_mapping():
    redis = FakeRedis()
    vector_db = VectorDB(dim=3, redis_client=redis)
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[None, FakeResult([type("TextRow", (), {"text_id": "task-123", "text": "doc-1"})()])])

    item_id = await vector_db.add(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        session=session,
        item_id="task-123",
        text="doc-1",
    )
    assert item_id == "task-123"
    assert await vector_db.save_to_redis() is True

    restored = VectorDB(dim=3, redis_client=redis)
    assert await restored.load_from_redis() is True

    results = await restored.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        session=session,
        top_k=1,
    )
    assert results == [
        {
            "text_id": "task-123",
            "text": "doc-1",
            "similarity": 1.0,
        }
    ]


@pytest.mark.asyncio
async def test_semantic_search_service_uses_json_cache_with_ttl():
    redis = FakeRedis()
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            None,
            None,
            FakeResult([type("TextRow", (), {"text_id": "task-123", "text": "doc-1"})()]),
        ]
    )
    vector_db = VectorDB(dim=3, redis_client=redis)
    service = SemanticSearchService(
        embedding_service=StubEmbeddingService(),
        vector_db=vector_db,
        redis_client=redis,
    )

    await service.index("doc-1", item_id="task-123", session=session)
    await service.index("doc-2", item_id="task-456", session=session)

    results = await service.search("query", top_k=1, session=session)

    assert results[0]["text_id"] == "task-123"
    cache_keys = [key for key in redis.store if key.startswith("semantic_search:")]
    assert len(cache_keys) == 1
    assert redis.ttl[cache_keys[0]] == 3600
    assert json.loads(redis.store[cache_keys[0]]) == results
