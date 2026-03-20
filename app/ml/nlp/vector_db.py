"""
Модуль для работы с векторной базой данных.
"""

from __future__ import annotations

from uuid import uuid4
import pickle

import faiss
import numpy as np
import redis.asyncio as redis

from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession

from ....app.models import Text


class VectorDB:
    """Класс для хранения эмбеддингов и поиска по ним с поддержкой Redis."""

    def __init__(self, dim: int, redis_client: redis.Redis | None = None):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.redis_client = redis_client
        self.index_key = "vector_db:faiss_index"
        self.ids_key = "vector_db:ids"

    async def add(
        self,
        embedding: list[float] | np.ndarray,
        session: AsyncSession,
        item_id: str | None = None,
        text: str = "",
    ) -> str:
        """Добавить эмбеддинг и similarity. Сохранить текст в базу данных."""
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Эмбеддинг должен быть одномерным вектором")
        if vector.shape[0] != self.dim:
            raise ValueError(f"Размерность эмбеддинга должна быть равна {self.dim}")

        resolved_item_id = item_id or str(uuid4())
        self.index.add(vector.reshape(1, -1))
        self.ids.append(resolved_item_id)
        
        await session.execute(
            insert(Text).values(text_id=resolved_item_id, text=text)
        )
        await session.commit()
        
        return resolved_item_id

    async def search(
        self,
        query_embedding: list[float] | np.ndarray,
        session: AsyncSession,
        top_k: int = 5,
    ) -> list[dict]:
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
            
            text = await session.execute(select(Text).where(Text.text_id == self.ids[idx]))
            text_result = text.scalar_one_or_none()
            
            results.append(
                {
                    "index": int(idx),
                    "id": self.ids[idx],
                    "text": text_result.text if text_result else "",
                    "similarity": float(similarity),
                }
            )

        return results

    async def save_to_redis(self) -> bool:
        """Сохранить индекс и метаданные в Redis."""
        if self.redis_client is None:
            return False

        try:
            async with self.redis_client.pipeline(transaction=True) as pipeline:
                await pipeline.set(self.index_key, faiss.serialize_index(self.index).tobytes())
                await pipeline.set(self.ids_key, pickle.dumps(self.ids))
                await pipeline.execute()
            return True
        except Exception:
            return False

    async def load_from_redis(self) -> bool:
        """Загрузить индекс и метаданные из Redis."""
        if self.redis_client is None:
            return False

        try:
            index_bytes = await self.redis_client.get(self.index_key)
            ids_bytes = await self.redis_client.get(self.ids_key)

            if index_bytes:
                self.index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
            if ids_bytes:
                self.ids = pickle.loads(ids_bytes)

            return bool(index_bytes or ids_bytes)
        except Exception:
            return False
