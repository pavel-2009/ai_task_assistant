"""Task management router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.core.dependencies import check_owner, get_task_or_404
from app.db import get_async_session
from app.db_models import Task, User
from app.ml.nlp.tasks import process_task_tags_and_embedding, update_recommendations_for_task
from app.ml.recsys.tasks import delete_task_interactions, process_task_interaction
from app.schemas import SuccessMessageResponse, TaskCreate, TaskGet, TaskStatusResponse, TaskUpdate
from app.services import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


def get_task_service() -> TaskService:
    return TaskService(
        process_task_tags_and_embedding=process_task_tags_and_embedding,
        update_recommendations_for_task=update_recommendations_for_task,
        process_task_interaction=process_task_interaction,
        delete_task_interactions=delete_task_interactions,
    )


@router.get("/", status_code=status.HTTP_200_OK, response_model=list[TaskGet])
async def get_tasks(session: AsyncSession = Depends(get_async_session)):
    return await get_task_service().list_tasks(session=session)


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=TaskGet)
async def create_task(
    task: TaskCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    return await get_task_service().create_task(session=session, payload=task, current_user=current_user)


@router.get("/{task_id}/tags_status", status_code=status.HTTP_200_OK, response_model=TaskStatusResponse)
async def check_tags_status(
    task_id: int = Path(...),
    session: AsyncSession = Depends(get_async_session),
):
    return await get_task_service().get_tags_status(session=session, task_id=task_id)


@router.get("/{task_id}", status_code=status.HTTP_200_OK, response_model=TaskGet)
async def get_task(
    task: Task = Depends(get_task_or_404),
    current_user: User = Depends(get_current_user),
):
    return await get_task_service().record_view(task=task, current_user=current_user)


@router.post("/{task_id}/like", status_code=status.HTTP_200_OK, response_model=SuccessMessageResponse)
async def like_task(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
):
    return await get_task_service().like_task(session=session, task_id=task_id, current_user=current_user)


@router.put("/{task_id}", status_code=status.HTTP_200_OK, response_model=TaskGet)
async def update_task(
    task_id: int = Path(...),
    task_update: TaskUpdate | None = None,
    session: AsyncSession = Depends(get_async_session),
    task: Task = Depends(check_owner),
):
    return await get_task_service().update_task(
        session=session,
        task_id=task_id,
        task=task,
        task_update=task_update,
    )


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: int = Path(...),
    session: AsyncSession = Depends(get_async_session),
    is_author: bool = Depends(check_owner),
):
    return await get_task_service().delete_task(session=session, task_id=task_id, is_author=is_author)
