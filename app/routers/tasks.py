"""
Роутер для управления задачами 
"""

from fastapi import APIRouter, status, Depends, HTTPException, Path, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

import typing

from app.db_models import Task, User
from app.schemas import TaskCreate, TaskGet, TaskUpdate
from app.db import get_async_session
from app.auth import get_current_user
from app.ml.nlp.tasks import process_task_tags_and_embedding, update_recommendations_for_task
from app.ml.recsys.tasks import process_task_interaction, delete_task_interactions
from app.core.dependencies import get_task_or_404, check_owner


router = APIRouter(
    prefix="/tasks",
    tags=["tasks"]
)



@router.get("/", status_code=status.HTTP_200_OK, description="Получение всех задач")
async def get_tasks(
    session: AsyncSession = Depends(get_async_session)
):
    """Получение всех задач"""

    task = await session.execute(select(Task))
    tasks = task.scalars().all()

    return [
        TaskGet(
            id=task.id,
            title=task.title,
            description=task.description,
            author_id=task.author_id
        )
        for task in tasks
    ]


@router.post("/", status_code=status.HTTP_201_CREATED, description="Создание задачи")
async def create_task(
    task: TaskCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    request: Request = None
):
    """Создание задачи"""
    
    task = Task(**task.model_dump())
    task.author_id = current_user.id
    task.tags = None  # Изначально теги не установлены, они будут обработаны в фоне
    
    
    session.add(task)
    await session.commit()

    await session.refresh(task)
    
    # Запускаем фоновую задачу для обработки тегов и эмбеддингов
    process_task_tags_and_embedding.delay(
        task_id=task.id,
        title=task.title,
        description=task.description
    )
    
    # Запускаем фоновую задачу для обработки взаимодействия пользователя с задачей (создание)
    process_task_interaction.delay(
        user_id=current_user.id,
        task_id=task.id,
        event_type="create",
        weight=1,
    )
    
    # Запускаем фоновую задачу для обновления рекомендаций для всех задач (включая новую)
    update_recommendations_for_task.delay(
        task_id=task.id,
    )

    return task


@router.get("/tasks/{task_id}/tags_status", status_code=status.HTTP_200_OK, description="Проверка статуса обработки тегов задачи")
async def check_tags_status(
    task_id: int = Path(...),
    session: AsyncSession = Depends(get_async_session)
):
    """Проверка статуса обработки тегов задачи"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )

    return {
        "tags": task.tags,
        "is_processing": task.tags is None
    }


@router.get("/{task_id}", status_code=status.HTTP_200_OK, description="Получение задачи по ID")
async def get_task(
    task: Task = Depends(get_task_or_404),
    current_user: User = Depends(get_current_user)
):
    """Получение задачи по ID"""

    # Запускаем фоновую задачу для обработки взаимодействия пользователя с задачей (просмотр)
    process_task_interaction.delay(
        user_id=current_user.id,
        task_id=task.id,
        event_type="view",
        weight=0.5,
    )
    
    return TaskGet(
        id=task.id,
        title=task.title,
        description=task.description,
        author_id=task.author_id
    )
    
    
@router.post("/{task_id}/like", status_code=status.HTTP_200_OK, description="Поставить лайк задаче")
async def like_task(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Поставить лайк задаче"""
    
    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
    
    # Запускаем фоновую задачу для обработки взаимодействия пользователя с задачей (лайк)
    process_task_interaction.delay(
        user_id=current_user.id,
        task_id=task_id,
        event_type="like",
        weight=1,
    )

    return {"message": "Задаче поставлен лайк"}


@router.put("/{task_id}", status_code=status.HTTP_200_OK, description="Обновление задачи")
async def update_task(
    task_id: int = Path(...),
    task_update: typing.Optional[TaskUpdate] = None,
    session: AsyncSession = Depends(get_async_session),
    task: Task = Depends(check_owner)
):
    """Обновление задачи"""

    update_dict = task_update.model_dump(exclude_unset=True) if task_update else {}
    
    if "title" in update_dict or "description" in update_dict:
        # Если обновляются title или description, нужно заново обработать теги и эмбеддинги
        process_task_tags_and_embedding.delay(
            task_id=task.id,
            title=update_dict.get("title", task.title),
            description=update_dict.get("description", task.description)
        )
        
        update_dict["tags"] = None  # Сбрасываем теги, они будут обновлены в фоне
        
        update_recommendations_for_task.delay(
            task_id=task.id
        )
        

    await session.execute(
        update(Task).where(Task.id == task_id).values(**update_dict)
    )
    
    await session.commit()

    return TaskGet(
        id=task.id,
        title=update_dict.get("title", task.title),
        description=update_dict.get("description", task.description),
        author_id=task.author_id,
        tags=update_dict.get("tags", task.tags)
    )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT, description="Удаление задачи")
async def delete_task(
    task_id: int = Path(...),
    request: Request = None,
    session: AsyncSession = Depends(get_async_session),
    task: Task = Depends(check_owner)
):
    """Удаление задачи"""

    # Удаляем вызовы взаимодействия текущей задачи
    delete_task_interactions.delay(
        task_id=task_id
    )

    await session.execute(
        delete(Task).where(Task.id == task_id)
    )
    
    # Удаляем данные из векторной базы и семантического поиска
    semantic_search_service = request.app.state.semantic_search_service
    await semantic_search_service.delete(item_id=str(task_id), session=session)
    
    await session.commit()
    
    return None
