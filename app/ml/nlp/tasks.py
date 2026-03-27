"""
Фоновые задачи для обработки текстов.
"""

from __future__ import annotations

from sqlalchemy import update, insert, select, create_engine
from sqlalchemy.orm import sessionmaker

import numpy as np

import json
import torch

from app.celery_app import celery_app
from app.services import get_ner, get_semantic_search, get_embedding, get_recsys_vector_db, get_image_embedding

from app.models import Task, Text
from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.recsys.vector_db.recsys_vector_db import RecSysVectorDB
from app.ml.cv.embedding.image_embedding_service import ImageEmbeddingService


sync_engine = create_engine("sqlite:///./test.db")
SyncSession = sessionmaker(sync_engine)


@celery_app.task(name="process_task_tags_and_embedding")
def process_task_tags_and_embedding(task_id: int, title: str, description: str):
    """Фоновая задача для обработки тегов и эмбеддингов задачи."""
    
    ner_service = get_ner()
    
    text = f"{title}\n{description}"
    tags_result = ner_service.tag_task(text)

    semantic_search_service = get_semantic_search()
    session = SyncSession()
    
    session.execute(
        update(Task).where(Task.id == task_id).values(tags=json.dumps(tags_result))
    )
    
    semantic_search_service.index_sync(
        text=text,
        session=session,
        item_id=task_id,
    )

    session.execute(
        insert(Text).values(
            item_id=task_id,
            text=text
        )
    )
    session.commit()
    session.close()
    
    
@celery_app.task(name="reindex_tasks")
def reindex_tasks():
    """Фоновая задача для реиндексации задач при старте приложения."""
    
    semantic_search_service = get_semantic_search()
    
    session = SyncSession()
    
    tasks = session.execute(select(Task.id, Task.title, Task.description))
    tasks = tasks.scalars().all()
    
    for task in tasks:
        text = f"{task.title}\n{task.description}"
        
        if task.id not in semantic_search_service.vector_db.ids:
            semantic_search_service.index_sync(
                text=text,
                session=session,
                item_id=task.id
            )
    session.close()
    
    
@celery_app.task(name="update_recommendations_for_task")
def update_recommendations_for_task(task_id: int):
    """Обновляет эмбеддинг задачи в FAISS и Redis."""
    # Получить задачу из БД
    session = SyncSession()
    task = session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()
    
    embedding_service: EmbeddingService = get_embedding()
    
    text = f"{task.title}\n{task.description}"
    avatar_file = task.avatar_file  # Путь к файлу аватара, если он есть
    
    if avatar_file:
        # Если есть аватар, нужно получить его эмбеддинг и объединить с текстовым
        image_embedding_service: ImageEmbeddingService = get_image_embedding()
        
        with open(avatar_file, "rb") as f:
            avatar_bytes = f.read()
        image_embedding = image_embedding_service.get_embedding(avatar_bytes)
        text_embedding = embedding_service.encode_one(text)
        
        # Объединяем эмбеддинги (например, конкатенацией)
        embedding = np.concatenate([text_embedding, image_embedding])
        
    else:

        embedding = embedding_service.encode_one(text)
        image_embedding = torch.zeros(512)  # Заполнитель для отсутствующего изображения, если размер эмбеддинга 896
        
        embedding = np.concatenate([embedding, image_embedding])
        
    # Нормализуем эмбеддинг
    embedding = embedding / np.linalg.norm(embedding)    
    
    # Добавить в vector_db (если нет) или обновить (если есть)
    rs_vector_db: RecSysVectorDB = get_recsys_vector_db()
    
    if rs_vector_db.ids_to_idx.get(str(task_id)) is not None:
        rs_vector_db.update(item_id=str(task_id), embedding=embedding)
        
    else: 
        rs_vector_db.add_vector(vector=embedding, task_id=str(task_id))
