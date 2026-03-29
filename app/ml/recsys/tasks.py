"""
Асинхронные задачи для рекомендательной системы.
"""

from app.celery_app import celery_app
from app.db_models import Interaction, Event

from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import redis

from sqlalchemy import select, update, create_engine, delete
from sqlalchemy.orm import sessionmaker

from datetime import datetime
import pickle


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
def train_collaborative_filtering_model(
    model: AlternatingLeastSquares,
    user_item_matrix: csr_matrix,
    redis_client: redis.Redis
):
    """Обучение модели коллаборативной фильтрации с сохранением модели в Redis."""

    model.fit(user_item_matrix)
        
    redis_client.set("collaborative_filtering_model", pickle.dumps((model.user_factors, model.item_factors)))
