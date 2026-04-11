"""
Сервис для семантического поиска по эмбеддингам.
"""

from __future__ import annotations

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config
from .embedding_service import EmbeddingService
from .vector_db import VectorDB


class SemanticSearchService:
    """Сервис для семантического поиска по эмбеддингам с кешированием в Redis."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_db: VectorDB | None = None,
        redis_client: redis.Redis | None = None,
    ) -> None:
        
        self.embedding_service = embedding_service
        self.redis_client = redis_client
        self.vector_db = vector_db or VectorDB(dim=embedding_service.dimension, redis_client=redis_client)

    @staticmethod
    def _normalize_text(text: str, field_name: str) -> str:
        if not isinstance(text, str):
            raise ValueError(f"{field_name} должен быть строкой")

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError(f"{field_name} не может быть пустым")

        return normalized_text

    async def index(self, text: str, session: AsyncSession, item_id: str | int | None = None) -> str:
        """Индексировать текст, добавляя его эмбеддинг в базу данных."""
        
        normalized_text = self._normalize_text(text, "Текст для индексирования")
        
        embedding = self.embedding_service.encode_one(normalized_text)
        
        resolved_item_id = await self.vector_db.add(
            embedding, session=session, item_id=item_id, text=normalized_text
        )
        
        await self.vector_db.save_to_redis()
        await self.clear_cache()
        
        return str(resolved_item_id)
    
    async def search(self, query: str, session: AsyncSession, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
        """Искать документы, наиболее похожие на запрос."""
        normalized_query = self._normalize_text(query, "Запрос")
        
        results = await self.vector_db.search(
            self.embedding_service.encode_one(normalized_query),
            session=session,
            top_k=top_k,
            query=normalized_query,
        )
        
        sorted_docs = sorted(results, key=lambda item: item["similarity"], reverse=True)[:top_k]
        return sorted_docs
    
    async def delete(self, item_id: str | int) -> None:
        """Удалить документ из базы данных и очистить кеш."""
        await self.vector_db.delete(str(item_id))
        await self.clear_cache()

    async def clear_cache(self) -> None:
        """Очистить весь кеш поиска."""
        await self.vector_db.clear_search_cache()

    async def save_index(self) -> bool:
        """Сохранить FAISS индекс в Redis."""
        return await self.vector_db.save_to_redis()

    async def load_index(self) -> bool:
        """Загрузить FAISS индекс из Redis."""
        return await self.vector_db.load_from_redis()
