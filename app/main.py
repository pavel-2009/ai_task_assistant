"""
Точка входа в приложение
"""

from fastapi import FastAPI, status, Path, HTTPException, Depends, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, update, delete, insert

import typing
import os
import uuid

from .models import TaskGet, TaskCreate, TaskUpdate, Task
from .db import get_async_session
from .utils.image_ops import validate_image, resize_image


# Создание приложения
app = FastAPI()


# === ЭНДПОИНТЫ ===
@app.get("/ping", status_code=status.HTTP_200_OK, description="Health-check эндпоинт")
async def ping():
    """Health-check"""

    return {
        "status": "OK"
    }


@app.get("/tasks", status_code=status.HTTP_200_OK, description="Получение всех задач")
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
            description=task.description
        )
        for task in tasks
    ]


@app.post("/tasks", status_code=status.HTTP_201_CREATED, description="Создание задачи")
async def create_task(
    task: TaskCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """Создание задачи"""
    
    task = Task(**task.model_dump())

    session.add(task)
    await session.commit()

    await session.refresh(task)

    return task


@app.get("/tasks/{task_id}", status_code=status.HTTP_200_OK, description="Получение задачи по ID")
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
            description=task.description
        )

    raise HTTPException(
        status_code=404,
        detail="Задача с указанным ID не найдена"
    )


@app.put("/tasks/{task_id}", status_code=status.HTTP_200_OK, description="Обновление задачи")
async def update_task(
    task_id: int = Path(...),
    task_update: typing.Optional[TaskUpdate] = None,
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


@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, description="Удаление задачи")
async def delete_task(
    task_id: int = Path(...),
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

    await session.execute(
        delete(Task).where(Task.id == task_id)
    )
    await session.commit()
    
    return None


@app.post("/tasks/{task_id}/avatar", status_code=status.HTTP_200_OK, description="Загрузка аватара для задачи")
async def upload_avatar(
    task_id: int = Path(...),
    image: UploadFile = None,
    session: AsyncSession = Depends(get_async_session)
):
    """Загрузка аватара для задачи"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Изображение не предоставлено"
        )

    with image.file as f:
        image_bytes = f.read()

    if not validate_image(image_bytes):
        raise HTTPException(
            status_code=400,
            detail="Невалидное изображение"
        )

    image = resize_image(image_bytes)

    if not os.path.exists("avatars"):
        os.makedirs("avatars")  

    filename = f"{task_id}_{uuid.uuid4().hex}.jpeg"

    with open(f"avatars/{filename}", "wb") as f:
        f.write(image)

    return {"filepath": f"avatars/{filename}", "filename": filename}