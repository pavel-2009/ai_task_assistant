"""
Модуль для работы с векторной базой данных. Здесь будет класс VectorDB, который будет хранить эмбеддинги и обеспечивать поиск по ним.
"""

import json

import faiss
import numpy as np
import redis


class VectorDB:
    """Класс для хранения эмбеддингов и поиска по ним с поддержкой Redis"""

    def __init__(self, dim: int, redis_client: redis.Redis | None = None):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts: list[str] = []  # Список текстов, соответствующих эмбеддингам в индексе
        self.redis_client = redis_client
        self.index_key = "vector_db:faiss_index"
        self.texts_key = "vector_db:texts"
        self.settings_key = "vector_db:settings"

    def _normalize_embedding(self, embedding: list[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Эмбеддинг должен быть одномерным непустым вектором")
        if vector.shape[0] != self.dim:
            raise ValueError(f"Ожидается размерность эмбеддинга {self.dim}, получено {vector.shape[0]}")
        return vector

    def add(self, embedding: list[float] | np.ndarray, text: str) -> None:
        """Добавить эмбеддинг и связанный с ним текст в базу данных"""
        if not text:
            raise ValueError("Текст не может быть пустым")

        vector = self._normalize_embedding(embedding)
        self.index.add(vector.reshape(1, -1))
        self.texts.append(text)

    def search(self, query_embedding: list[float] | np.ndarray, top_k: int = 5) -> list[dict]:
        """Поиск наиболее похожих текстов по эмбеддингу запроса"""
        if top_k <= 0:
            raise ValueError("top_k должен быть больше 0")
        if self.index.ntotal == 0 or len(self.texts) == 0:
            raise ValueError("База данных документов пуста. Индексируйте документы перед поиском.")

        query_vector = self._normalize_embedding(query_embedding).reshape(1, -1)
        current_top_k = min(top_k, len(self.texts))

        distances, indices = self.index.search(query_vector, current_top_k)

        results: list[dict] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.texts):
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
        """Сохранить индекс и тексты в Redis"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.set(self.index_key, faiss.serialize_index(self.index).tobytes())
            self.redis_client.set(self.texts_key, json.dumps(self.texts, ensure_ascii=False))
            self.redis_client.set(self.settings_key, json.dumps({"dim": self.dim}, ensure_ascii=False))
            return True
        except Exception as e:
            print(f"Ошибка при сохранении индекса в Redis: {e}")
            return False

    def load_from_redis(self) -> bool:
        """Загрузить индекс и тексты из Redis"""
        if not self.redis_client:
            return False
        try:
            settings_raw = self.redis_client.get(self.settings_key)
            if settings_raw:
                settings = json.loads(settings_raw)
                saved_dim = int(settings.get("dim", self.dim))
                if saved_dim != self.dim:
                    raise ValueError(f"Размерность индекса в Redis ({saved_dim}) не совпадает с ожидаемой ({self.dim})")

            index_bytes = self.redis_client.get(self.index_key)
            texts_raw = self.redis_client.get(self.texts_key)

            if index_bytes is None or texts_raw is None:
                return False

            self.index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
            self.texts = list(json.loads(texts_raw))
            return True
        except Exception as e:
            print(f"Ошибка при загрузке индекса из Redis: {e}")
            return False
