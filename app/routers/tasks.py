"""
Роутер для управления задачами 
"""

from fastapi import APIRouter, status, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from app.models import Task, TaskGet, TaskCreate, TaskUpdate, User
from app.db import get_async_session
from app.auth import get_current_user

import typing


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
    session: AsyncSession = Depends(get_async_session)
):
    """Создание задачи"""
    
    task = Task(**task.model_dump())
    task.author_id = current_user.id

    session.add(task)
    await session.commit()

    await session.refresh(task)

    return task


@router.get("/{task_id}", status_code=status.HTTP_200_OK, description="Получение задачи по ID")
async def get_task(
    task_id: int = Path(...),
    session: AsyncSession = Depends(get_async_session)
):
    """Получение задачи по ID"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is not None:
        return TaskGet(
            id=task.id,
            title=task.title,
            description=task.description,
            author_id=task.author_id
        )

    raise HTTPException(
        status_code=404,
        detail="Задача с указанным ID не найдена"
    )


@router.put("/{task_id}", status_code=status.HTTP_200_OK, description="Обновление задачи")
async def update_task(
    task_id: int = Path(...),
    task_update: typing.Optional[TaskUpdate] = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Обновление задачи"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
    
    if task.author_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="У вас нет прав на изменение этой задачи"
        )

    update_dict = task_update.model_dump(exclude_unset=True) if task_update else {}

    await session.execute(
        update(Task).where(Task.id == task_id).values(**update_dict)
    )
    await session.commit()

    return TaskGet(
        id=task.id,
        title=update_dict.get("title", task.title),
        description=update_dict.get("description", task.description)
    )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT, description="Удаление задачи")
async def delete_task(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Удаление задачи"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )

    if task.author_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="У вас нет прав на удаление этой задачи"
        )

    await session.execute(
        delete(Task).where(Task.id == task_id)
    )
    await session.commit()
    
    return None