import pytest
from pydantic import ValidationError
from types import SimpleNamespace

from app.routers import tasks as tasks_router
from app.schemas.task import TaskCreate


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class DummySession:
    def __init__(self):
        self.added = []
        self.committed = False
        self.refreshed = []

    def add(self, obj):
        # имитируем автоинкремент id после add/commit
        obj.id = 101
        self.added.append(obj)

    async def commit(self):
        self.committed = True

    async def refresh(self, obj):
        self.refreshed.append(obj)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_task_success(monkeypatch):
    session = DummySession()
    current_user = SimpleNamespace(id=7)

    calls = {"tags": None, "interaction": None, "recs": None}

    def _tags_delay(**kwargs):
        calls["tags"] = kwargs

    def _interaction_delay(**kwargs):
        calls["interaction"] = kwargs

    def _recs_delay(**kwargs):
        calls["recs"] = kwargs

    monkeypatch.setattr(tasks_router.process_task_tags_and_embedding, "delay", _tags_delay)
    monkeypatch.setattr(tasks_router.process_task_interaction, "delay", _interaction_delay)
    monkeypatch.setattr(tasks_router.update_recommendations_for_task, "delay", _recs_delay)

    payload = TaskCreate(title="Task title", description="Task description")

    result = await tasks_router.create_task(task=payload, current_user=current_user, session=session)

    assert result.id == 101
    assert result.author_id == 7
    assert result.tags is None

    assert session.committed is True
    assert len(session.added) == 1
    assert len(session.refreshed) == 1

    assert calls["tags"] == {
        "task_id": 101,
        "title": "Task title",
        "description": "Task description",
    }
    assert calls["interaction"] == {
        "user_id": 7,
        "task_id": 101,
        "event_type": "create",
        "weight": 1,
    }
    assert calls["recs"] == {"task_id": 101}


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        {"title": "", "description": "desc"},  # пустой title
        {"title": "x" * 21, "description": "desc"},  # слишком длинный title
        {"title": "ok", "description": "x" * 401},  # слишком длинный description
    ],
)
def test_task_create_validation_errors(payload):
    with pytest.raises(ValidationError):
        TaskCreate(**payload)


@pytest.mark.unit
def test_task_create_requires_title():
    with pytest.raises(ValidationError):
        TaskCreate(description="only description")
