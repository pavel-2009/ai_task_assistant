from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.core.dependencies import check_owner, get_task_or_404


@pytest.mark.asyncio
async def test_get_task_or_404_exists():
    task = SimpleNamespace(id=1, author_id=10)
    result_mock = SimpleNamespace(scalar_one_or_none=lambda: task)
    session = AsyncMock()
    session.execute.return_value = result_mock

    result = await get_task_or_404(task_id=1, session=session)
    assert result.id == 1


@pytest.mark.asyncio
async def test_get_task_or_404_not_exists():
    result_mock = SimpleNamespace(scalar_one_or_none=lambda: None)
    session = AsyncMock()
    session.execute.return_value = result_mock

    with pytest.raises(HTTPException) as exc:
        await get_task_or_404(task_id=1, session=session)

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_check_owner_owner():
    task = SimpleNamespace(author_id=10)
    current_user = SimpleNamespace(id=10)
    result = await check_owner(task=task, current_user=current_user)
    assert result is task


@pytest.mark.asyncio
async def test_check_owner_not_owner():
    task = SimpleNamespace(author_id=10)
    current_user = SimpleNamespace(id=20)
    with pytest.raises(HTTPException) as exc:
        await check_owner(task=task, current_user=current_user)

    assert exc.value.status_code == 403
