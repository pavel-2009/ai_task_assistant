"""
Фоновые задачи для обработки текстов.
"""

from __future__ import annotations

from sqlalchemy import update, insert, select, create_engine
from sqlalchemy.orm import sessionmaker

import json

from app.celery_app import celery_app
from app.services import get_ner, get_semantic_search

from app.models import Task, Text


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
