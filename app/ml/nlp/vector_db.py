"""
Модуль для работы с векторной базой данных.
"""

from __future__ import annotations

from uuid import uuid4
import pickle
import asyncio

import faiss
import numpy as np
import redis.asyncio as redis

from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Text


class VectorDB:
    """Класс для хранения эмбеддингов и поиска по ним с поддержкой Redis."""

    def __init__(
        self,
        dim: int,
        redis_client: redis.Redis | None = None
    ) -> None:
        
        self.dim = dim # Размерность эмбеддингов
        self.index = faiss.IndexFlatIP(dim) # Индекс для поиска по косинусной близости (нормализованные векторы)
        self.ids: list[str] = []
        self.redis_client = redis_client
        self.index_key = "vector_db:faiss_index"
        self.ids_key = "vector_db:ids"
        self._lock = asyncio.Lock() # Асинхронный лок для обеспечения потокобезопасности при добавлении и поиске

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
        
        async with self._lock: # Блокируем для обеспечения потокобезопасности при добавлении в индекс и обновлении ids
            try:
                self.index.add(vector.reshape(1, -1)) # -1 - для сохранения размерности (1, dim)
                self.ids.append(resolved_item_id)
            except Exception as e:
                raise RuntimeError(f"Ошибка при добавлении эмбеддинга в индекс: {e}")
        
        await session.execute(
            insert(Text).values(text_id=resolved_item_id, text=text)
        )
        
        return resolved_item_id
    
    async def add_batch(
        self,
        embeddings: list[list[float]] | np.ndarray,
        session: AsyncSession,
        item_ids: list[str] | None = None,
        texts: list[str] | None = None
    ) -> list[str]:
        """Добавить несколько эмбеддингов и сохранить тексты в базу данных."""
        
        vectors = np.asarray(embeddings, dtype=np.float32)
        
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Эмбеддинги должны быть двумерным массивом с размерностью {self.dim}")

        batch_size = vectors.shape[0]
        
        # Валидация item_ids
        if item_ids is not None and len(item_ids) != batch_size:
            raise ValueError(f"Длина item_ids ({len(item_ids)}) не совпадает с количеством эмбеддингов ({batch_size})")
        
        # Валидация texts
        if texts is not None and len(texts) != batch_size:
            raise ValueError(f"Длина texts ({len(texts)}) не совпадает с количеством эмбеддингов ({batch_size})")

        resolved_item_ids = item_ids or [str(uuid4()) for _ in range(batch_size)]
        
        async with self._lock:
            try:
                self.index.add(vectors)
                self.ids.extend(resolved_item_ids)
            except Exception as e:
                raise RuntimeError(f"Ошибка при добавлении эмбеддингов в индекс: {e}")
        
        if texts:
            await session.execute(
                insert(Text).values([
                    {"text_id": item_id, "text": text} for item_id, text in zip(resolved_item_ids, texts)
                ])
            )
        
        return resolved_item_ids

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
        
        async with self._lock:
            if not self.ids:
                return []
            
            # Проверка консистентности между индексом и ids
            if self.index.ntotal != len(self.ids):
                raise RuntimeError(f"Консистентность нарушена: размер индекса ({self.index.ntotal}) != размер ids ({len(self.ids)})")

            limit = min(top_k, len(self.ids))
            similarities, indices = self.index.search(vector.reshape(1, -1), limit)

        results: list[dict] = []
        
        # Фильтруем только валидные индексы (>=0 и < len(self.ids))
        valid_entries = [
            (idx, sim) for idx, sim in zip(indices[0], similarities[0])
            if idx >= 0 and idx < len(self.ids)
        ]
        
        if valid_entries:
            text_ids = [self.ids[idx] for idx, _ in valid_entries]
            texts = await session.execute(select(Text).where(Text.text_id.in_(text_ids)))
            
            text_map = {text.text_id: text.text for text in texts.scalars().all()}
            
            for idx, sim in valid_entries:
                text_id = self.ids[idx]
                results.append({"text_id": text_id, "similarity": float(sim), "text": text_map.get(text_id, "")})

        return results

    async def save_to_redis(self) -> bool:
        """Сохранить индекс и метаданные в Redis."""
        if self.redis_client is None:
            return False

        try:
            async with self._lock:
                index_data = faiss.serialize_index(self.index).tobytes()
                ids_data = pickle.dumps(self.ids)
            
            async with self.redis_client.pipeline(transaction=True) as pipeline:
                await pipeline.set(self.index_key, index_data)
                await pipeline.set(self.ids_key, ids_data)
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

            async with self._lock:
                if index_bytes:
                    self.index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
                if ids_bytes:
                    self.ids = pickle.loads(ids_bytes)

            return bool(index_bytes or ids_bytes)
        except Exception:
            return False
        
    async def delete(
        self, 
        item_id: str
    ) -> None:
        """Удалить документ из базы данных и очистить кеш."""
        
        # Удаление из векторной базы и кеша
        async with self._lock:
            if item_id in self.ids:
                idx = self.ids.index(item_id)
                self.index.remove_ids(np.array([idx], dtype=np.int64))
                self.ids.pop(idx)
        
        await self.clear_cache()
