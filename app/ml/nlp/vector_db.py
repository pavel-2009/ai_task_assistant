"""
Модуль для работы с векторной базой данных.
"""

from __future__ import annotations

import pickle
from uuid import uuid4

import faiss
import numpy as np
from redis import Redis


class VectorDB:
    """Класс для хранения эмбеддингов и поиска по ним с поддержкой Redis."""

    def __init__(self, dim: int, redis_client: Redis | None = None):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.redis_client = redis_client
        self.index_key = "vector_db:faiss_index"
        self.ids_key = "vector_db:ids"

    def add(
        self,
        embedding: list[float] | np.ndarray,
        item_id: str | None = None,
    ) -> str:
        """Добавить эмбеддинг и similarity."""
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Эмбеддинг должен быть одномерным вектором")
        if vector.shape[0] != self.dim:
            raise ValueError(f"Размерность эмбеддинга должна быть равна {self.dim}")

        resolved_item_id = item_id or str(uuid4())
        self.index.add(vector.reshape(1, -1))
        self.ids.append(resolved_item_id)
        return resolved_item_id

    def search(self, query_embedding: list[float] | np.ndarray, top_k: int = 5) -> list[dict]:
        """Поиск наиболее похожих текстов по эмбеддингу запроса."""
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Эмбеддинг запроса не может быть пустым")
        if vector.shape[0] != self.dim:
            raise ValueError(f"Размерность эмбеддинга должна быть равна {self.dim}")
        if not self.ids:
            return []

        limit = min(top_k, len(self.ids))
        similarities, indices = self.index.search(vector.reshape(1, -1), limit)

        results: list[dict] = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "index": int(idx),
                    "id": self.ids[idx],
                    "similarity": float(similarity),
                }
            )

        return results

    def save_to_redis(self) -> bool:
        """Сохранить индекс и метаданные в Redis."""
        if self.redis_client is None:
            return False

        try:
            pipeline = self.redis_client.pipeline()
            pipeline.set(self.index_key, faiss.serialize_index(self.index).tobytes())
            pipeline.set(self.ids_key, pickle.dumps(self.ids))
            pipeline.execute()
            return True
        except Exception:
            return False

    def load_from_redis(self) -> bool:
        """Загрузить индекс и метаданные из Redis."""
        if self.redis_client is None:
            return False

        try:
            index_bytes = self.redis_client.get(self.index_key)
            ids_bytes = self.redis_client.get(self.ids_key)

            if index_bytes:
                self.index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
            if ids_bytes:
                self.ids = pickle.loads(ids_bytes)

            return bool(index_bytes or ids_bytes)
        except Exception:
            return False
