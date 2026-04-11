"""
Модуль для работы с векторной базой данных.
"""

from __future__ import annotations

import hashlib
import json
from uuid import uuid4
import pickle
import asyncio

import faiss
import numpy as np
import redis.asyncio as redis

from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config
from app.db_models import Text


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
        self.id_to_position: dict[str, int] = {} # Словарь для быстрого поиска индекса по item_id
        self.redis_client = redis_client
        self.index_key = "vector_db:faiss_index"
        self.ids_key = "vector_db:ids"
        self.delete_cache_prefix = "vector_db:delete_cache:"
        self.search_cache_prefix = "vector_db:search_cache:"
        self._lock = asyncio.Lock() # Асинхронный лок для обеспечения потокобезопасности при добавлении и поиске

    async def add(
        self,
        embeddings: list[float] | list[list[float]] | np.ndarray,
        session: AsyncSession,
        item_id: str | list[str] | None = None,
        text: str | list[str] | None = None,
    ) -> str | list[str]:
        """Добавить один или несколько эмбеддингов и сохранить тексты в базу данных."""

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] != self.dim:
            raise ValueError(f"Эмбеддинги должны быть непустым массивом формы (n, {self.dim})")

        batch_size = vectors.shape[0]

        item_ids = [item_id] if isinstance(item_id, str) else item_id
        texts = [text] if isinstance(text, str) else text

        if item_ids is not None and len(item_ids) != batch_size:
            raise ValueError(f"Длина item_id ({len(item_ids)}) не совпадает с количеством эмбеддингов ({batch_size})")
        if texts is not None and len(texts) != batch_size:
            raise ValueError(f"Длина text ({len(texts)}) не совпадает с количеством эмбеддингов ({batch_size})")

        resolved_item_ids = [str(value) for value in item_ids] if item_ids is not None else [str(uuid4()) for _ in range(batch_size)]
        resolved_texts = texts or [""] * batch_size

        async with self._lock:
            try:
                self.index.add(vectors)
                self.ids.extend(resolved_item_ids)
            except Exception as e:
                raise RuntimeError(f"Ошибка при добавлении эмбеддингов в индекс: {e}")
        
        if any(resolved_texts):
            await session.execute(
                insert(Text).values([
                    {"text_id": current_id, "text": current_text}
                    for current_id, current_text in zip(resolved_item_ids, resolved_texts)
                ])
            )

        start_index = len(self.ids) - batch_size
        self.id_to_position.update(
            {current_id: start_index + idx for idx, current_id in enumerate(resolved_item_ids)}
        )

        return resolved_item_ids[0] if batch_size == 1 else resolved_item_ids
    
    async def search(
        self,
        query_embedding: list[float] | np.ndarray,
        session: AsyncSession,
        top_k: int = config.DEFAULT_TOP_K,
        query: str | None = None,
    ) -> list[dict]:
        """Поиск наиболее похожих текстов по эмбеддингу запроса."""

        cache_key = self._build_search_cache_key(query, top_k)
        cached_result = await self._get_from_cache(cache_key) if cache_key else None
        if cached_result is not None:
            return cached_result

        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or vector.shape[0] != self.dim:
            raise ValueError(f"Эмбеддинг запроса должен быть непустым вектором размерности {self.dim}")
        
        async with self._lock:
            if not self.ids:
                return []
            
            # Проверка консистентности между индексом и ids
            if self.index.ntotal != len(self.ids):
                raise RuntimeError(f"Консистентность нарушена: размер индекса ({self.index.ntotal}) != размер ids ({len(self.ids)})")

            limit = min(top_k, len(self.ids))
            similarities, indices = self.index.search(vector.reshape(1, -1), limit)

        results: list[dict] = []
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(self.ids)]
        valid_index_set = set(valid_indices)
        if valid_indices:
            text_ids = [self.ids[idx] for idx in valid_indices]
            texts = await session.execute(select(Text).where(Text.text_id.in_(text_ids)))
            text_map = {text.text_id: text.text for text in texts.scalars().all()}

            for idx, sim in zip(indices[0], similarities[0]):
                if idx not in valid_index_set:
                    continue

                text_id = self.ids[idx]
                text_value = text_map.get(text_id, "")
                title, _, description = text_value.partition("\n")
                score = float(sim)
                results.append(
                    {
                        "text_id": text_id,
                        "similarity": score,
                        "text": text_value,
                        "task_id": int(text_id) if text_id.isdigit() else None,
                        "title": title or None,
                        "description": description or None,
                        "score": score,
                    }
                )

        if cache_key and results:
            await self._save_to_cache(cache_key, results)
        return results

    async def save_to_redis(self) -> bool:
        """Сохранить индекс и метаданные в Redis."""
        if self.redis_client is None:
            return False

        try:
            async with self._lock:
                index_data = faiss.serialize_index(self.index)
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
                    self.index = faiss.deserialize_index(index_bytes)
                if ids_bytes:
                    self.ids = pickle.loads(ids_bytes)
                self.id_to_position = {current_id: idx for idx, current_id in enumerate(self.ids)}

            return bool(index_bytes or ids_bytes)
        except Exception:
            return False
        
    async def delete(
        self, 
        item_id: str
    ) -> None:
        """Удалить документ из индекса, сохранив остальные записи через кэш Redis."""

        async with self._lock:
            if item_id not in self.id_to_position:
                return

            retained_ids = [current_id for current_id in self.ids if current_id != item_id]
            retained_vectors: dict[str, np.ndarray] = {
                current_id: self.index.reconstruct(idx)
                for idx, current_id in enumerate(self.ids)
                if current_id != item_id
            }

            if self.redis_client is not None:
                for current_id in retained_ids:
                    cache_key = f"{self.delete_cache_prefix}{current_id}"
                    cached_vector = await self.redis_client.get(cache_key)
                    
                    if cached_vector is None:
                        await self.redis_client.set(cache_key, pickle.dumps(retained_vectors[current_id]))

            self.index.reset() # Сброс индекса, так как FAISS не поддерживает удаление отдельных векторов
            self.ids = retained_ids
            self.id_to_position = {current_id: idx for idx, current_id in enumerate(self.ids)}

            vectors_to_restore: list[np.ndarray] = []
            keys_to_cleanup: list[str] = []

            for current_id in self.ids:
                if self.redis_client is not None:
                    
                    cache_key = f"{self.delete_cache_prefix}{current_id}"
                    cached_vector = await self.redis_client.get(cache_key)
                    
                    if cached_vector is not None:
                        vectors_to_restore.append(pickle.loads(cached_vector))
                        keys_to_cleanup.append(cache_key)
                        
                        continue
                    
                vectors_to_restore.append(retained_vectors[current_id])

            if vectors_to_restore:
                self.index.add(np.asarray(vectors_to_restore, dtype=np.float32))

            if self.redis_client is not None and keys_to_cleanup:
                await self.redis_client.delete(*keys_to_cleanup)

    def _build_search_cache_key(self, query: str | None, top_k: int) -> str | None:
        """Собрать ключ кеша для поиска по запросу."""
        if query is None:
            return None
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"{self.search_cache_prefix}{query_hash}:{top_k}"

    async def _get_from_cache(self, cache_key: str) -> list[dict] | None:
        """Получить результаты поиска из кеша Redis."""
        if self.redis_client is None:
            return None

        try:
            cached = await self.redis_client.get(cache_key)
            return json.loads(cached) if cached else None
        except Exception:
            return None

    async def _save_to_cache(self, cache_key: str, results: list[dict], ttl: int = 3600) -> None:
        """Сохранить результаты поиска в кеш Redis."""
        if self.redis_client is None:
            return

        try:
            await self.redis_client.setex(cache_key, ttl, json.dumps(results, ensure_ascii=False))
        except Exception:
            return

    async def clear_search_cache(self) -> None:
        """Очистить весь кеш поиска."""
        if self.redis_client is None:
            return

        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=f"{self.search_cache_prefix}*")
                if keys:
                    await self.redis_client.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            return
        
