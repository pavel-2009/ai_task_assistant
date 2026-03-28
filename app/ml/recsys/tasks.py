"""
Асинхронные задачи для рекомендательной системы.
"""

from app.celery_app import celery_app
from app.db_models import Interaction, Event

from sqlalchemy import select, update, create_engine, delete
from sqlalchemy.orm import sessionmaker

from datetime import datetime


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
        select(Interaction).where(Interaction.task_id == task_id)  
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
