"""
Асинхронные задачи для рекомендательной системы.
"""

import logging

from app.celery_app import celery_app
from app.celery_metrics import track_celery_task
from app.db_models import Interaction, Event

from implicit.als import AlternatingLeastSquares

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from datetime import datetime
import pickle
import asyncio

from app.services import get_collaborative_filtering_recommender
from app.db import async_session


logger = logging.getLogger(__name__)


@celery_app.task(name="process_task_interaction")
def process_task_interaction(
    user_id: int,
    task_id: int,
    event_type: str,
    weight: int
):
    """Обработка взаимодействия пользователя с задачей для рекомендательной системы."""
    asyncio.run(_process_task_interaction_async(user_id, task_id, event_type, weight))


async def _process_task_interaction_async(
    user_id: int,
    task_id: int,
    event_type: str,
    weight: int
):
    """Асинхронная реализация обработки взаимодействия."""
    
    async with async_session() as session:
        result = await session.execute(
            select(Interaction).where(
                Interaction.task_id == task_id, 
                Interaction.user_id == user_id
            )
        )
        
        if not result.scalar_one_or_none():
            interaction = Interaction(
                user_id=user_id,
                task_id=task_id,
                event_type=Event(event_type),
                weight=weight,
                created_at=datetime.utcnow()
            )
            session.add(interaction)
            
        else:
            await session.execute(
                update(Interaction)
                .where(
                    Interaction.task_id == task_id, 
                    Interaction.user_id == user_id
                )
                .values(
                    event_type=Event(event_type),
                    weight=weight,
                    created_at=datetime.utcnow()
                )
            )
            
        await session.commit()
    

@celery_app.task(name="delete_task_interactions")   
def delete_task_interactions(task_id: int):
    """Удаление всех взаимодействий для задачи при ее удалении."""
    asyncio.run(_delete_task_interactions_async(task_id))


async def _delete_task_interactions_async(task_id: int):
    """Асинхронная реализация удаления взаимодействий."""
    
    async with async_session() as session:
        await session.execute(
            delete(Interaction).where(Interaction.task_id == task_id)
        )
        
        await session.commit() 
    
    
@celery_app.task(name="train_collaborative_filtering_model")
@track_celery_task("train_collaborative_filtering_model")
def train_collaborative_filtering_model():
    """Обучение модели коллаборативной фильтрации с сохранением модели в Redis."""
    try:
        asyncio.run(_train_collaborative_filtering_model_async())
    except RuntimeError as e:
        logger.warning(f"Could not train collaborative filtering model (service may not be initialized): {e}")
    except Exception as e:
        logger.error(f"Error during collaborative filtering model training: {e}", exc_info=True)


async def _train_collaborative_filtering_model_async():
    """Асинхронная реализация обучения модели коллаборативной фильтрации."""
    
    try:
        collaborative_filtering_recommender = get_collaborative_filtering_recommender()
    except RuntimeError as e:
        logger.warning(f"Collaborative filtering recommender service not initialized: {e}")
        return

    model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
    
    async with async_session() as session:
        (
            matrix,
            user_to_idx,
            idx_to_task,
            unique_users,
            unique_tasks
        ) = await collaborative_filtering_recommender.build_user_item_matrix(session)
        
        model.fit(matrix)
        
        # Вычленяем самые популярные задачи для новых пользователей (холодный старт)
        result = await session.execute(
            select(Interaction.task_id)
            .group_by(Interaction.task_id)
            .order_by(func.count(Interaction.task_id).desc())
            .limit(10)
        )
        popular_tasks = [task[0] for task in result.all()]

        
        redis_client = collaborative_filtering_recommender.redis_client
            
        redis_client.set("collaborative_filtering_model", pickle.dumps((matrix, user_to_idx, idx_to_task, unique_users, unique_tasks, popular_tasks)))  # Сериализация модели в Redis
