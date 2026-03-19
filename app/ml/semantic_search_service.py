"""
Сервис для семантического поиска по эмбеддингам. Использует модель SentenceTransformer для получения эмбеддингов текстов и вычисления сходства между ними.
"""

import json
import hashlib
import redis.asyncio as redis
from .embedding_service import EmbeddingService
from .vector_db import VectorDB


class SemanticSearchService:
    """Сервис для семантического поиска по эмбеддингам с кешированием в Redis"""
    
    def __init__(self, embedding_service: EmbeddingService, database: VectorDB = None, redis_client: redis.Redis = None):
        self.embedding_service: EmbeddingService = embedding_service
        if database is None:
            database = VectorDB(dim=384, redis_client=redis_client)
        self.database = database
        self.redis_client = redis_client
        self.cache_prefix = "semantic_search:"
        self.index_key = "semantic_search:faiss_index"
        self.texts_key = "semantic_search:texts"

    def index(self, text: str):
        """Индексировать текст, добавляя его эмбеддинг в базу данных"""
        
        if not text or len(text) == 0:
            raise ValueError("Текст для индексирования не может быть пустым")
        
        embedding = self.embedding_service.encode_one(text)
        self.database.add(embedding, text)
    
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Искать документы, наиболее похожие на запрос"""
        
        if not query or len(query) == 0:
            raise ValueError("Запрос не может быть пустым")
        
        # Проверяем кеш в Redis
        cache_key = self._get_cache_key(query, top_k)
        if self.redis_client:
            try:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    return cached_result
            except Exception as e:
                print(f"Ошибка при работе с Redis кешем: {e}")
        
        query_embedding = self.embedding_service.encode_one(query)
        
        if not self.database:
            raise ValueError("База данных документов пуста. Индексируйте документы перед поиском.")
        
        results = self.database.search(query_embedding, top_k=top_k)
        
        # Сортируем документы по сходству в убывающем порядке (выше similarity = первым)
        sorted_docs = sorted(results, key=lambda x: x["similarity"], reverse=True)
        sorted_docs = sorted_docs[:top_k]
        
        # Сохраняем результат в Redis кеш
        if self.redis_client:
            try:
                self._save_to_cache(cache_key, sorted_docs)
            except Exception as e:
                print(f"Ошибка при сохранении в Redis кеш: {e}")
        
        return sorted_docs
    
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Генерируем уникальный ключ для кеша"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self.cache_prefix}{query_hash}:{top_k}"
    
    
    def _get_from_cache(self, cache_key: str) -> list[dict] | None:
        """Получить результаты из Redis кеша (синхронный вызов)"""
        if not self.redis_client:
            return None
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
        return None
    
    
    def _save_to_cache(self, cache_key: str, results: list[dict], ttl: int = 3600) -> None:
        """Сохранить результаты в Redis кеш с TTL (1 час по умолчанию)"""
        if not self.redis_client:
            return
        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(results))
        except Exception:
            pass
    
    
    def clear_cache(self) -> None:
        """Очистить весь кеш поиска"""
        if not self.redis_client:
            return
        try:
            # Удаляем все ключи с префиксом
            keys = self.redis_client.keys(f"{self.cache_prefix}*")
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