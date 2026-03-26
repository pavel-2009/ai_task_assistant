"""
Векторная база данных для эмбеддингов задач. Использует Faiss для хранения и поиска векторов задач.
"""

import numpy as np
import faiss
import redis


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
        
    
    def add_vector(self, vector: np.ndarray, task_id: str) -> None:
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
        
        
    def search(self, vector: np.ndarray, top_k: int = 5) -> list[str]:
        """Ищем похожие векторы изображений и возвращаем их IDs."""
        if self.redis_client:
            # Загружаем индекс и IDs из Redis
            index_data = self.redis_client.get(self.index_key)
            ids_data = self.redis_client.get(self.ids_key)
            if index_data and ids_data:
                self.index = faiss.deserialize_index(index_data)
                self.ids = ids_data.decode().split(",")
                self.ids_to_idx = {task_id: idx for idx, task_id in enumerate(self.ids)}
        
        if len(self.ids) == 0:
            return []
        
        _, I = self.index.search(vector.reshape(1, -1), top_k)
        similar_ids = [self.ids[i] for i in I[0] if i < len(self.ids)]
        
        return similar_ids
            