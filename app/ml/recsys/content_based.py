"""
Сервис для работы с контентной рекомендательной системой. В данном случае - для получения эмбеддингов изображений и текстовых описаний, которые затем можно использовать для поиска похожих задач.
"""

import numpy as np

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pathlib import Path
import redis

from app.ml.cv.embedding.image_embedding_service import ImageEmbeddingService
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.recsys.vector_db.recsys_vector_db import RecSysVectorDB
from app.db_models import Task


class ContentBasedRecommender:
    """Сервис для контентной рекомендательной системы."""
    
    
    def __init__(
        self,
        image_embedding_service: ImageEmbeddingService = None,
        text_embedding_service: EmbeddingService = None,
        image_vector_db: RecSysVectorDB = None,
        redis_client: redis.Redis = None
    ):
        self.image_embedding_service: ImageEmbeddingService = image_embedding_service or ImageEmbeddingService()
        self.text_embedding_service: EmbeddingService = text_embedding_service or EmbeddingService()
        self.recsys_vector_db: RecSysVectorDB = image_vector_db or RecSysVectorDB(dim=896, redis_client=redis_client)


    async def _get_image_embedding(self, image: str):
        """Получаем эмбеддинг для изображения."""
        
        # Проверяем кэш
        cache_key = f"img_emb:{image}"
        cached_emb = await self.recsys_vector_db.redis_client.get(cache_key)
        if cached_emb is not None:
            return np.frombuffer(cached_emb, dtype=np.float32)
        
        image_path = Path(image)
        
        if not image_path.is_file():
            raise ValueError(f"Файл изображения {image} не найден")
        
        with image_path.open("rb") as f:
            image_bytes = f.read()
            
        # Добавляем в кэш
        embedding = self.image_embedding_service.get_embedding(image_bytes)
        await self.recsys_vector_db.redis_client.set(cache_key, embedding.tobytes(), ex=86400)  # Кэшируем на 24 часа
        
        return embedding
    
    
    async def _get_text_embedding(self, text: str):
        """Получаем эмбеддинг для текстового описания."""
        return self.text_embedding_service.get_embedding(text)
    
    
    async def _get_task_embedding(self, image: str, text: str):
        """Получаем объединенный эмбеддинг для задачи на основе изображения и текста."""
        
        # Проверяем кэш
        cache_key = f"task_emb:{image}:{text}"
        cached_emb = await self.recsys_vector_db.redis_client.get(cache_key)
        
        if cached_emb is not None:
            return np.frombuffer(cached_emb, dtype=np.float32)
        
        image_emb = await self._get_image_embedding(image)
        text_emb = await self._get_text_embedding(text)
        
        # Объединяем эмбеддинги (можно использовать разные методы, например, конкатенацию или усреднение)
        combined_emb = np.concatenate([image_emb, text_emb])
        
        # Сохраняем в кэш
        await self.recsys_vector_db.redis_client.set(cache_key, combined_emb.tobytes(), ex=86400)  # Кэшируем на 24 часа
        
        return combined_emb
    
    
    async def _get_task(
        self,
        task_id: int,
        session: AsyncSession
    ) -> dict:
        """Получаем данные задачи из базы данных."""
        
        task = await session.execute(select(Task).where(Task.id == task_id))
        task = task.scalar_one_or_none()
        
        if not task:
            raise ValueError(f"Задача с ID {task_id} не найдена")
        
        return {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "avatar_file": task.avatar_file,
            "tags": task.tags
        }
    
    async def _find_similar_tasks(
        self,
        task_embedding: np.ndarray,
        session: AsyncSession,
        top_k: int = 5,
        author_id: int = None
    ) -> list[dict]:
        """Находим похожие задачи на основе эмбеддингов."""
        
        # Ищем похожие задачи в векторной базе данных изображений
        tasks = await self.recsys_vector_db.search(task_embedding, top_k=top_k)
        if not tasks:
            return []
        
        # Получаем данные похожих задач из базы данных
        similar_tasks = []
        
        tasks = await session.execute(select(Task).where(Task.id.in_(tasks)))
        tasks = tasks.scalars().all()
        
        for task, score in tasks:
            if author_id is not None:
                if task.author_id != author_id:
                    continue
                similar_tasks.append({
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "avatar_file": task.avatar_file,
                    "tags": task.tags,
                    "similarity_score": score
                })     
                   
        return similar_tasks


    async def recommend(
        self,
        task_id: int,
        session: AsyncSession,
        author_id: int = None
    ) -> list[dict]:
        """Рекомендуем похожие задачи на основе эмбеддингов."""

        task = await self._get_task(task_id, session)
        
        # Получаем эмбеддинг для текущей задачи
        task_emb = await self._get_task_embedding(
            image=task["avatar_file"],  # Предполагается, что avatar_file содержит путь к изображению
            text=task["description"]  # Предполагается, что description содержит текстовое описание
        )
        
        # Находим похожие задачи
        similar_tasks = await self._find_similar_tasks(task_emb, session, top_k=10, author_id=author_id)
        
        return similar_tasks
        