"""
Асинхронные задачи для рекомендательной системы.
"""

from app.celery_app import celery_app
from app.db_models import Interaction, Event

from implicit.als import AlternatingLeastSquares

from sqlalchemy import select, update, create_engine, delete, func
from sqlalchemy.orm import sessionmaker

from datetime import datetime
import pickle

from app.services import get_collaborative_filtering_recommender


sync_engine = create_engine("sqlite:///./test.db")
SyncSession = sessionmaker(sync_engine)


@celery_app.task(name="process_task_interaction")
def process_task_interaction(
    user_id: int,
    task_id: int,
    event_type: str,
    weight: int
):
    """Обработка взаимодействия пользователя с задачей для рекомендательной системы."""
    
    session = SyncSession()
    
    task = session.execute(
        select(Interaction).where(Interaction.task_id == task_id, Interaction.user_id == user_id    )  
    )
    
    if not task.scalar_one_or_none():
        interaction = Interaction(
            user_id=user_id,
            task_id=task_id,
            event_type=Event(event_type),
            weight=weight,
            created_at=datetime.utcnow()
        )
        session.add(interaction)
        
    else:
        session.execute(
            update(Interaction)
            .where(Interaction.task_id == task_id, Interaction.user_id == user_id)
            .values(
                event_type=Event(event_type),
                weight=weight,
                created_at=datetime.utcnow()
            )
        )
        
    session.commit()
    

@celery_app.task(name="delete_task_interactions")   
def delete_task_interactions(task_id: int):
    """Удаление всех взаимодействий для задачи при ее удалении."""
    
    session = SyncSession()
    
    session.execute(
        delete(Interaction).where(Interaction.task_id == task_id)
    )
    
    session.commit() 
    
    
@celery_app.task(name="train_collaborative_filtering_model")
def train_collaborative_filtering_model():
    """Обучение модели коллаборативной фильтрации с сохранением модели в Redis."""

    model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
    
    collaborative_filtering_recommender = get_collaborative_filtering_recommender()
    session = SyncSession()
    
    user_factors, item_factors, user_to_idx, task_to_idx, idx_to_task = collaborative_filtering_recommender.build_user_item_matrix(session)
    
    model.fit(user_factors)
    
    # Вычленяем самые популярные задачи для новых пользователей (холодный старт)
    popular_tasks = session.execute(
        select(Interaction.task_id).group_by(Interaction.task_id).order_by(func.count(Interaction.task_id).desc()).limit(10)
    )
    popular_tasks = [task[0] for task in popular_tasks.scalars().all()]

    
    redis_client = collaborative_filtering_recommender.redis_client
        
    redis_client.set("collaborative_filtering_model", pickle.dumps((user_factors, item_factors, user_to_idx, task_to_idx, idx_to_task, popular_tasks)))  # Сериализация модели в Redis
