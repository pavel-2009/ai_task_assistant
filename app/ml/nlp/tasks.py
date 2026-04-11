"""
Фоновые задачи для обработки текстов.
"""

from __future__ import annotations

import logging

from sqlalchemy import update, select

import numpy as np

import json
import asyncio

from app.celery_app import celery_app
from app.celery_metrics import track_celery_task
from app.services import get_ner, get_semantic_search, get_embedding, get_recsys_vector_db, get_image_embedding
from app.db import async_session

from app.db_models import Task


logger = logging.getLogger(__name__)
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.vector_db import VectorDB
from app.ml.cv.embedding.image_embedding_service import ImageEmbeddingService


@celery_app.task(name="process_task_tags_and_embedding")
def process_task_tags_and_embedding(task_id: int, title: str, description: str):
    """Фоновая задача для обработки тегов и эмбеддингов задачи."""
    asyncio.run(_process_task_tags_and_embedding_async(task_id, title, description))


async def _process_task_tags_and_embedding_async(task_id: int, title: str, description: str):
    """Асинхронная реализация обработки тегов и эмбеддингов."""
    try:
        ner_service = get_ner()
        semantic_search_service = get_semantic_search()
    except RuntimeError as e:
        logger.warning("NLP services are not initialized: %s", e)
        return
    
    text = f"{title}\n{description}"
    tags_result = ner_service.tag_task(text)
    
    async with async_session() as session:
        await session.execute(
            update(Task).where(Task.id == task_id).values(tags=json.dumps(tags_result))
        )
        
        await semantic_search_service.index(
            text=text,
            session=session,
            item_id=str(task_id),
        )
        await session.commit()
    
    
@celery_app.task(name="reindex_tasks")
@track_celery_task("reindex_tasks")
def reindex_tasks():
    """Фоновая задача для реиндексации задач при старте приложения."""
    try:
        asyncio.run(_reindex_tasks_async())
    except RuntimeError as e:
        logger.warning(f"Could not reindex tasks (service may not be initialized): {e}")
    except Exception as e:
        logger.error(f"Error during task reindexing: {e}", exc_info=True)


async def _reindex_tasks_async():
    """Асинхронная реализация реиндексации задач."""
    
    try:
        semantic_search_service = get_semantic_search()
    except RuntimeError as e:
        logger.warning(f"Semantic search service not initialized: {e}")
        return
    
    async with async_session() as session:
        result = await session.execute(select(Task.id, Task.title, Task.description))
        tasks = result.all()
        
        for task in tasks:
            text = f"{task.title}\n{task.description}"
            
            task_id = str(task.id)
            if task_id not in semantic_search_service.vector_db.ids:
                await semantic_search_service.index(
                    text=text,
                    session=session,
                    item_id=task_id
                )
    
    
@celery_app.task(name="update_recommendations_for_task")
def update_recommendations_for_task(task_id: int):
    """Обновляет эмбеддинг задачи в FAISS и Redis."""
    asyncio.run(_update_recommendations_for_task_async(task_id))


async def _update_recommendations_for_task_async(task_id: int):
    """Асинхронная реализация обновления рекомендаций."""
    try:
        embedding_service: EmbeddingService = get_embedding()
        rs_vector_db: VectorDB = get_recsys_vector_db()
    except RuntimeError as e:
        logger.warning("Recommendation services are not initialized: %s", e)
        return
    
    async with async_session() as session:
        # Получить задачу из БД
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()
        
        if not task:
            return
        
        text = f"{task.title}\n{task.description}"
        avatar_file = task.avatar_file  # Путь к файлу аватара, если он есть
        
        text_embedding = embedding_service.encode_one(text).astype(np.float32)
        image_embedding = np.zeros(512, dtype=np.float32)

        if avatar_file:
            # Если есть аватар, нужно получить его эмбеддинг и объединить с текстовым
            image_embedding_service: ImageEmbeddingService = get_image_embedding()

            with open(avatar_file, "rb") as f:
                avatar_bytes = f.read()
            image_embedding = image_embedding_service.get_embedding(avatar_bytes).astype(np.float32)

        # Всегда собираем единый вектор в формате [text(384) + image(512)]
        embedding = np.concatenate([text_embedding, image_embedding]).astype(np.float32)
            
        # Нормализуем эмбеддинг
        embedding = embedding / np.linalg.norm(embedding)    
        
        # Добавить в vector_db (если нет) или обновить (если есть)
        item_id = str(task_id)
        if rs_vector_db.id_to_position.get(item_id) is not None:
            await rs_vector_db.delete(item_id=item_id)

        await rs_vector_db.add(
            embeddings=embedding,
            session=session,
            item_id=item_id,
        )


@celery_app.task(name="warmup_llm", bind=True, max_retries=10)
@track_celery_task("warmup_llm")
def warmup_llm(self):
    """Фоновая задача для прогрева LLM модели с повторами."""
    
    retry_delay = 30  # секунд
    
    try:
        from app.services import get_llm
        llm_service = get_llm()
        asyncio.run(llm_service.warmup())
        logger.info("LLM warmup completed successfully")
    except Exception as exc:
        logger.warning("LLM warmup attempt failed: %s. Retry in %d seconds...", exc, retry_delay)
        # Используем встроенный механизм retry Celery
        raise self.retry(exc=exc, countdown=retry_delay)
        
