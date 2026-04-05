from __future__ import annotations

import numpy as np


class DummyDelayTask:
    def delay(self, *args, **kwargs):
        return {"queued": True, "args": args, "kwargs": kwargs}


class DummyEmbeddingService:
    model = object()
    dimension = 4

    def encode_one(self, text: str) -> np.ndarray:
        return np.array([len(text), 1.0, 0.0, 0.5], dtype=float)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([[len(t), 1.0, 0.0, 0.5] for t in texts], dtype=float)


class DummySemanticSearchService:
    cache_prefix = "test"

    def __init__(self):
        self.embedding_service = object()
        self.vector_db = object()
        self._indexed: list[str] = []

    async def search(self, query: str, session, top_k: int = 5):
        return [{"id": i + 1, "text": text} for i, text in enumerate(self._indexed[:top_k])]

    async def index(self, text: str, session):
        self._indexed.append(text)


class DummyNerService:
    is_ready = True

    def tag_task(self, text: str):
        tags = []
        for token in ("python", "fastapi", "redis", "celery", "postgres"):
            if token in text.lower():
                tags.append((token, 0.99))
        return {"technologies": tags}


class DummyRagService:
    async def ask(self, query: str, session, top_k: int, use_cache: bool):
        return {
            "answer": f"answer for {query}",
            "sources": [1, 2],
            "confidence": 0.9,
            "cached": use_cache,
        }

    async def ask_stream(self, query: str, session, top_k: int):
        for token in ["hello", "world"]:
            yield token
