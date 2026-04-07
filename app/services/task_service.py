"""Task business logic separated from HTTP routers."""

from __future__ import annotations

from fastapi import HTTPException, Response, status
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_models import Task, User
from app.schemas import SuccessMessageResponse, TaskCreate, TaskGet, TaskStatusResponse, TaskUpdate


class TaskService:
    """CRUD helpers for tasks and their async side effects."""

    def __init__(
        self,
        process_task_tags_and_embedding,
        update_recommendations_for_task,
        process_task_interaction,
        delete_task_interactions,
    ) -> None:
        self.process_task_tags_and_embedding = process_task_tags_and_embedding
        self.update_recommendations_for_task = update_recommendations_for_task
        self.process_task_interaction = process_task_interaction
        self.delete_task_interactions = delete_task_interactions

    async def list_tasks(self, session: AsyncSession) -> list[TaskGet]:
        result = await session.execute(select(Task))
        return [self._to_schema(task) for task in result.scalars().all()]

    async def create_task(self, session: AsyncSession, payload: TaskCreate, current_user: User) -> TaskGet:
        task = Task(**payload.model_dump())
        task.author_id = current_user.id
        task.tags = None

        session.add(task)
        await session.commit()
        await session.refresh(task)

        self.process_task_tags_and_embedding.delay(task_id=task.id, title=task.title, description=task.description)
        self.process_task_interaction.delay(user_id=current_user.id, task_id=task.id, event_type="create", weight=1)
        self.update_recommendations_for_task.delay(task_id=task.id)

        return self._to_schema(task)

    async def get_tags_status(self, session: AsyncSession, task_id: int) -> TaskStatusResponse:
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()
        if task is None:
            raise HTTPException(status_code=404, detail="Task with the specified ID was not found")

        return TaskStatusResponse(tags=task.tags, is_processing=task.tags is None)

    async def record_view(self, task: Task, current_user: User) -> TaskGet:
        self.process_task_interaction.delay(user_id=current_user.id, task_id=task.id, event_type="view", weight=0.5)
        return self._to_schema(task)

    async def like_task(
        self,
        session: AsyncSession,
        task_id: int,
        current_user: User,
    ) -> SuccessMessageResponse:
        result = await session.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()
        if task is None:
            raise HTTPException(status_code=404, detail="Task with the specified ID was not found")

        self.process_task_interaction.delay(user_id=current_user.id, task_id=task_id, event_type="like", weight=1)
        return SuccessMessageResponse(message="Task has been liked")

    async def update_task(
        self,
        session: AsyncSession,
        task_id: int,
        task: Task,
        task_update: TaskUpdate | None = None,
    ) -> TaskGet:
        update_dict = task_update.model_dump(exclude_unset=True) if task_update else {}

        if "title" in update_dict or "description" in update_dict:
            self.process_task_tags_and_embedding.delay(
                task_id=task.id,
                title=update_dict.get("title", task.title),
                description=update_dict.get("description", task.description),
            )
            update_dict["tags"] = None
            self.update_recommendations_for_task.delay(task_id=task.id)

        await session.execute(update(Task).where(Task.id == task_id).values(**update_dict))
        await session.commit()

        return TaskGet(
            id=task.id,
            title=update_dict.get("title", task.title),
            description=update_dict.get("description", task.description),
            author_id=task.author_id,
            tags=update_dict.get("tags", task.tags),
        )

    async def delete_task(self, session: AsyncSession, task_id: int, is_author: bool) -> Response:
        if not is_author:
            raise HTTPException(status_code=403, detail="You do not have permission to delete this task")

        self.delete_task_interactions.delay(task_id=task_id)
        await session.execute(delete(Task).where(Task.id == task_id))
        await session.commit()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @staticmethod
    def _to_schema(task: Task) -> TaskGet:
        return TaskGet(
            id=task.id,
            title=task.title,
            description=task.description,
            author_id=task.author_id,
            tags=task.tags,
        )
