from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.security import hash_password
from app.db_models import Task, User
from app.schemas import TaskCreate, TaskUpdate, UserCreate
from app.services.auth_service import AuthService
from app.services.task_service import TaskService
from app.services.user_service import UserService


class SpyDelayTask:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def delay(self, **kwargs):
        self.calls.append(kwargs)
        return {"queued": True}


@pytest.mark.asyncio
async def test_auth_service_register_and_login(unit_session_maker):
    service = AuthService()

    async with unit_session_maker() as session:
        created = await service.register_user(
            session=session,
            user_payload=UserCreate(username="service_user", password="Password1!"),
        )

        token = await service.login_user(
            session=session,
            username="service_user",
            password="Password1!",
        )

    assert created.username == "service_user"
    assert token.token_type == "bearer"
    assert token.access_token


@pytest.mark.asyncio
async def test_auth_service_duplicate_registration_returns_400(unit_session_maker):
    service = AuthService()

    async with unit_session_maker() as session:
        payload = UserCreate(username="dup_user", password="Password1!")
        await service.register_user(session=session, user_payload=payload)

        with pytest.raises(HTTPException) as exc_info:
            await service.register_user(session=session, user_payload=payload)

    assert exc_info.value.status_code == 400
    assert "already exists" in exc_info.value.detail


@pytest.mark.asyncio
async def test_user_service_get_by_username(unit_session_maker):
    service = UserService()

    async with unit_session_maker() as session:
        user = User(username="lookup_user", password=hash_password("Password1!"))
        session.add(user)
        await session.commit()

        found = await service.get_by_username(session=session, username="lookup_user")
        missing = await service.get_by_username(session=session, username="unknown")

    assert found is not None
    assert found.username == "lookup_user"
    assert missing is None


@pytest.mark.asyncio
async def test_task_service_create_update_and_delete(unit_session_maker):
    tags_task = SpyDelayTask()
    recs_task = SpyDelayTask()
    interaction_task = SpyDelayTask()
    delete_interactions_task = SpyDelayTask()

    service = TaskService(
        process_task_tags_and_embedding=tags_task,
        update_recommendations_for_task=recs_task,
        process_task_interaction=interaction_task,
        delete_task_interactions=delete_interactions_task,
    )

    async with unit_session_maker() as session:
        user = User(username="task_owner", password=hash_password("Password1!"))
        session.add(user)
        await session.commit()
        await session.refresh(user)

        created = await service.create_task(
            session=session,
            payload=TaskCreate(title="TaskSvc", description="Service layer test"),
            current_user=user,
        )

        task = await session.get(Task, created.id)
        updated = await service.update_task(
            session=session,
            task_id=created.id,
            task=task,
            task_update=TaskUpdate(title="TaskSvcNew", description="Updated text"),
        )

        response = await service.delete_task(session=session, task_id=created.id, is_author=True)

    assert created.author_id == user.id
    assert updated.title == "TaskSvcNew"
    assert response.status_code == 204
    assert tags_task.calls
    assert recs_task.calls
    assert any(call["event_type"] == "create" for call in interaction_task.calls)
    assert delete_interactions_task.calls[0]["task_id"] == created.id


@pytest.mark.asyncio
async def test_task_service_delete_forbidden_for_non_author(unit_session_maker):
    service = TaskService(SpyDelayTask(), SpyDelayTask(), SpyDelayTask(), SpyDelayTask())

    async with unit_session_maker() as session:
        with pytest.raises(HTTPException) as exc_info:
            await service.delete_task(session=session, task_id=1, is_author=False)

    assert exc_info.value.status_code == 403
