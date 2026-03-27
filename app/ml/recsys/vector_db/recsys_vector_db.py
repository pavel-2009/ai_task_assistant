"""
Векторная база данных для эмбеддингов задач. Использует Faiss для хранения и поиска векторов задач.
"""

import numpy as np
import faiss
import redis

import pickle


class RecSysVectorDB:
    """Класс для хранения эмбеддингов задач."""

    def __init__(self, dim: int, redis_client: redis.Redis | None = None) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self.ids_to_idx: dict[str, int] = {}
        self.redis_client = redis_client
        self.index_key = "recsys_vector_db:faiss_index"
        self.ids_key = "recsys_vector_db:ids"
        self.delete_cache_prefix = "recsys_vector_db:delete_cache:"
        
    
    async def add_vector(self, vector: np.ndarray, task_id: str) -> None:
        """Добавляем вектор задачи в базу данных."""
        if task_id in self.ids_to_idx:
            raise ValueError(f"Задача с ID {task_id} уже существует в базе данных.")
        
        idx = len(self.ids)
        self.index.add(vector.reshape(1, -1))
        self.ids.append(task_id)
        self.ids_to_idx[task_id] = idx
        
        # Сохраняем индекс и IDs в Redis
        if self.redis_client:
            self.redis_client.set(self.index_key, faiss.serialize_index(self.index))
            self.redis_client.set(self.ids_key, ",".join(self.ids))
        
        
    async def search(self, vector: np.ndarray, top_k: int = 5) -> list[str]:
        """Ищем похожие векторы изображений и возвращаем их IDs."""
        if self.redis_client:
            # Загружаем индекс и IDs из Redis
            index_data = await self.redis_client.get(self.index_key)
            ids_data = await self.redis_client.get(self.ids_key)
            if index_data and ids_data:
                self.index = faiss.deserialize_index(index_data)
                self.ids = ids_data.decode().split(",")
                self.ids_to_idx = {task_id: idx for idx, task_id in enumerate(self.ids)}
        
        if len(self.ids) == 0:
            return []
        
        _, I = self.index.search(vector.reshape(1, -1), top_k)
        similar_ids = [self.ids[i] for i in I[0] if i < len(self.ids)]
        
        return similar_ids
    
    
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
        """Удалить документ из индекса, сохранив остальные записи через кэш Redis."""

        async with self._lock:
            if item_id not in self.ids_to_idx:
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
            self.ids_to_idx = {current_id: idx for idx, current_id in enumerate(self.ids)}

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
                
                
    async def update(
        self, 
        item_id: str, 
        embedding: np.ndarray
    ) -> None:
        """Обновить вектор задачи в базе данных."""
        if item_id not in self.ids_to_idx:
            raise ValueError(f"Задача с ID {item_id} не найдена в базе данных.")
        
        idx = self.ids_to_idx[item_id]
        self.index.reconstruct(idx)  # Восстанавливаем текущий вектор (необходимо для обновления)
        self.index.remove_ids(np.array([idx]))  # Удаляем старый вектор
        self.index.add(embedding.reshape(1, -1))  # Добавляем новый вектор
        