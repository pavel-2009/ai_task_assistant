"""
Фоновые задачи для обработки текстов.
"""

from __future__ import annotations

from sqlalchemy import update, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from ....app.celery_app import celery_app, get_ner_service, get_semantic_search_service

from app.ml.nlp.ner_service import NerService
from app.ml.nlp.semantic_search_service import SemanticSearchService
from app.db import get_async_session

from app.models import Task, Text


@celery_app.task(name="process_task_tags_and_embedding")
def process_task_tags_and_embedding(task_id: int, title: str, description: str):
    """Фоновая задача для обработки тегов и эмбеддингов задачи."""
    
    ner_service: NerService = get_ner_service()
    
    text = f"{title}\n{description}"
    tags_result = ner_service.tag_task(text)

    semantic_search_service: SemanticSearchService = get_semantic_search_service()
    session: AsyncSession = get_async_session()
    
    session.execute(
        update(Task).where(Task.id == task_id).values(tags=tags_result)
    )
    
    semantic_search_service.index(
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
    
    
@celery_app.task(name="reindex_tasks")
async def reindex_tasks():
    """Фоновая задача для реиндексации задач при старте приложения."""
    
    semantic_search_service = get_semantic_search_service()
    
    tasks = await get_async_session().execute(
        select(Task)
    )
    tasks = tasks.scalars().all()
    
    for task in tasks:
        text = f"{task.title}\n{task.description}"
        
        if task.id not in semantic_search_service.vector_db.ids:
            await semantic_search_service.index(
                text=text,
                session=get_async_session(),
                item_id=task.id
            )
