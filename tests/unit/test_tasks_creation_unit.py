"""Простые unit-тесты валидации схемы создания задач."""

import pytest
from pydantic import ValidationError

from app.schemas.task import TaskCreate


def test_task_create_valid_payload():
    payload = TaskCreate(title="Task", description="Description")
    assert payload.title == "Task"


@pytest.mark.parametrize(
    "data",
    [
        {"title": "", "description": "desc"},
        {"title": "x" * 21, "description": "desc"},
        {"title": "ok", "description": "x" * 401},
    ],
)
def test_task_create_invalid_payload(data):
    # Ключевой комментарий: проверяем только pydantic-валидацию без работы с БД.
    with pytest.raises(ValidationError):
        TaskCreate(**data)
