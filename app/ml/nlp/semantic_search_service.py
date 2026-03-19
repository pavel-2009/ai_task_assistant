"""
Сервис для семантического поиска по эмбеддингам. Использует модель SentenceTransformer для получения эмбеддингов текстов и вычисления сходства между ними.
"""

import hashlib
import json

import redis

from .embedding_service import EmbeddingService
from .vector_db import VectorDB


class SemanticSearchService:
    """Сервис для семантического поиска по эмбеддингам с кешированием в Redis"""

    def __init__(self, embedding_service: EmbeddingService, database: VectorDB | None = None, redis_client: redis.Redis | None = None):
        self.embedding_service: EmbeddingService = embedding_service
        if database is None:
            database = VectorDB(dim=384, redis_client=redis_client)
        elif database.redis_client is None:
            database.redis_client = redis_client

        self.database = database
        self.redis_client = redis_client
        self.cache_prefix = "semantic_search:"

    def index(self, text: str) -> None:
        """Индексировать текст, добавляя его эмбеддинг в базу данных"""
        if not text or len(text) == 0:
            raise ValueError("Текст для индексирования не может быть пустым")

        embedding = self.embedding_service.encode_one(text)
        self.database.add(embedding, text)
        self.clear_cache()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Искать документы, наиболее похожие на запрос"""
        if not query or len(query) == 0:
            raise ValueError("Запрос не может быть пустым")
        if top_k <= 0:
            raise ValueError("top_k должен быть больше 0")

        cache_key = self._get_cache_key(query, top_k)
        if self.redis_client:
            try:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    return cached_result
            except Exception as e:
                print(f"Ошибка при работе с Redis кешем: {e}")

        query_embedding = self.embedding_service.encode_one(query)
        results = self.database.search(query_embedding, top_k=top_k)
        normalized_results = self._normalize_results(results)

        if self.redis_client:
            try:
                self._save_to_cache(cache_key, normalized_results)
            except Exception as e:
                print(f"Ошибка при сохранении в Redis кеш: {e}")

        return normalized_results

    def _normalize_results(self, results: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for result in results:
            normalized.append(
                {
                    "index": int(result["index"]),
                    "text": str(result["text"]),
                    "similarity": float(result["similarity"]),
                }
            )
        return normalized

    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Генерируем уникальный ключ для кеша"""
        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        return f"{self.cache_prefix}{query_hash}:{top_k}"

    def _get_from_cache(self, cache_key: str) -> list[dict] | None:
        """Получить результаты из Redis кеша"""
        if not self.redis_client:
            return None

        cached = self.redis_client.get(cache_key)
        if cached is None:
            return None

        return self._normalize_results(json.loads(cached))

    def _save_to_cache(self, cache_key: str, results: list[dict], ttl: int = 3600) -> None:
        """Сохранить результаты в Redis кеш с TTL (1 час по умолчанию)"""
        if not self.redis_client:
            return

        payload = json.dumps(self._normalize_results(results), ensure_ascii=False)
        self.redis_client.setex(cache_key, ttl, payload)

    def clear_cache(self) -> None:
        """Очистить весь кеш поиска"""
        if not self.redis_client:
            return
        try:
            keys = list(self.redis_client.scan_iter(f"{self.cache_prefix}*"))
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            print(f"Ошибка при очистке кеша: {e}")

    def save_index(self) -> bool:
        """Сохранить FAISS индекс в Redis"""
        return self.database.save_to_redis()

    def load_index(self) -> bool:
        """Загрузить FAISS индекс из Redis"""
        return self.database.load_from_redis()
