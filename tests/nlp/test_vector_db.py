import numpy as np

from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.ml.nlp.vector_db import VectorDB


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttl = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value
        return True

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttl[key] = ttl
        return True

    def scan_iter(self, match=None):
        prefix = match[:-1] if match and match.endswith('*') else match
        for key in self.store:
            if prefix is None or key.startswith(prefix):
                yield key

    def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)
            self.ttl.pop(key, None)
        return len(keys)

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    def set(self, key, value):
        self.commands.append((key, value))
        return self

    def execute(self):
        for key, value in self.commands:
            self.redis.set(key, value)
        return True


class StubEmbeddingService:
    dimension = 3

    def encode_one(self, text: str) -> np.ndarray:
        base = {
            'doc-1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'doc-2': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'query': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
        return base[text]


def test_vector_db_returns_index_to_id_mapping_and_persists_it():
    redis = FakeRedis()
    vector_db = VectorDB(dim=3, redis_client=redis)

    item_id = vector_db.add(np.array([1.0, 0.0, 0.0], dtype=np.float32), 'doc-1', item_id='task-123')
    assert item_id == 'task-123'
    assert vector_db.save_to_redis() is True

    restored = VectorDB(dim=3, redis_client=redis)
    assert restored.load_from_redis() is True

    results = restored.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert results == [
        {
            'index': 0,
            'id': 'task-123',
            'text': 'doc-1',
            'similarity': 1.0,
        }
    ]


def test_semantic_search_service_uses_json_cache_with_ttl():
    redis = FakeRedis()
    vector_db = VectorDB(dim=3, redis_client=redis)
    service = SemanticSearchService(
        embedding_service=StubEmbeddingService(),
        vector_db=vector_db,
        redis_client=redis,
    )

    service.index('doc-1', item_id='task-123')
    service.index('doc-2', item_id='task-456')

    results = service.search('query', top_k=1)

    assert results[0]['id'] == 'task-123'
    cache_keys = list(redis.scan_iter(match='semantic_search:*'))
    assert len(cache_keys) == 1
    assert redis.ttl[cache_keys[0]] == 3600
    assert redis.get(cache_keys[0]).startswith('[{"index": 0, "id": "task-123"')
