"""
Модуль для работы с векторной базой данных.
"""

from __future__ import annotations

import pickle

import faiss
import numpy as np
from redis import Redis


class VectorDB:
    """Класс для хранения эмбеддингов и поиска по ним с поддержкой Redis."""

    def __init__(self, dim: int, redis_client: Redis | None = None):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts: list[str] = []
        self.redis_client = redis_client
        self.index_key = "vector_db:faiss_index"
        self.texts_key = "vector_db:texts"

    def add(self, embedding: list[float] | np.ndarray, text: str) -> None:
        """Добавить эмбеддинг и связанный с ним текст в базу данных."""
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Эмбеддинг должен быть одномерным вектором")
        if vector.shape[0] != self.dim:
            raise ValueError(f"Размерность эмбеддинга должна быть равна {self.dim}")

        self.index.add(vector.reshape(1, -1))
        self.texts.append(text)

    def search(self, query_embedding: list[float] | np.ndarray, top_k: int = 5) -> list[dict]:
        """Поиск наиболее похожих текстов по эмбеддингу запроса."""
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Эмбеддинг запроса не может быть пустым")
        if vector.shape[0] != self.dim:
            raise ValueError(f"Размерность эмбеддинга должна быть равна {self.dim}")
        if not self.texts:
            return []

        limit = min(top_k, len(self.texts))
        distances, indices = self.index.search(vector.reshape(1, -1), limit)

        results: list[dict] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            similarity = float(1.0 / (1.0 + float(dist)))
            results.append(
                {
                    "index": int(idx),
                    "text": self.texts[idx],
                    "similarity": similarity,
                }
            )

        return results

    def save_to_redis(self) -> bool:
        """Сохранить индекс и тексты в Redis."""
        if self.redis_client is None:
            return False

        try:
            pipeline = self.redis_client.pipeline()
            pipeline.set(self.index_key, faiss.serialize_index(self.index).tobytes())
            pipeline.set(self.texts_key, pickle.dumps(self.texts))
            pipeline.execute()
            return True
        except Exception:
            return False

    def load_from_redis(self) -> bool:
        """Загрузить индекс и тексты из Redis."""
        if self.redis_client is None:
            return False

        try:
            index_bytes = self.redis_client.get(self.index_key)
            texts_bytes = self.redis_client.get(self.texts_key)

            if index_bytes:
                self.index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
            if texts_bytes:
                self.texts = pickle.loads(texts_bytes)

            return bool(index_bytes or texts_bytes)
        except Exception:
            return False
