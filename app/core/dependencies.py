"""Зависимости приложения."""

from fastapi import Depends, HTTPException, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_async_session
from app.db_models import Task, User
from app.auth import get_current_user


async def get_task_or_404(task_id: int = Path(...), session: AsyncSession = Depends(get_async_session)) -> Task:
    """Получение задачи по ID или возврат 404 ошибки, если задача не найдена."""
    result = await session.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task


async def check_owner(task: Task = Depends(get_task_or_404), current_user: User = Depends(get_current_user)) -> Task:
    """Проверка, что текущий пользователь является владельцем задачи."""
    if task.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="У вас нет прав на эту операцию")
    return task 
