import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

from app.routers import avatars as avatars_router
from app.db_models import Task


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class DummyAvatarSession:
    def __init__(self, task):
        self.task = task
        self.executed = []
        self.committed = False

    async def execute(self, query):
        query_str = str(query)
        self.executed.append(query_str)
        if "SELECT" in query_str.upper():
            return _ScalarResult(self.task)
        return None

    async def commit(self):
        self.committed = True


def _make_upload_file(valid: bool = True) -> UploadFile:
    if valid:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        data = encoded.tobytes()
    else:
        data = b"not-an-image"

    return UploadFile(filename="avatar.jpg", file=io.BytesIO(data))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_avatar_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    task = Task(id=12, title="t", description="d", author_id=42)
    session = DummyAvatarSession(task)
    user = SimpleNamespace(id=42)

    delayed = {}

    def _delay(**kwargs):
        delayed.update(kwargs)

    monkeypatch.setattr(avatars_router.update_recommendations_for_task, "delay", _delay)

    response = await avatars_router.upload_avatar(
        task_id=12,
        image=_make_upload_file(valid=True),
        current_user=user,
        session=session,
    )

    assert response.filepath.startswith("avatars/")
    assert response.filename.endswith(".jpeg")
    assert (tmp_path / response.filepath).exists()
    assert session.committed is True
    assert delayed == {"task_id": 12}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_avatar_not_found():
    session = DummyAvatarSession(task=None)

    with pytest.raises(HTTPException) as exc:
        await avatars_router.upload_avatar(
            task_id=999,
            image=_make_upload_file(valid=True),
            current_user=SimpleNamespace(id=1),
            session=session,
        )

    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_avatar_forbidden_owner_mismatch():
    task = Task(id=13, title="t", description="d", author_id=2)
    session = DummyAvatarSession(task=task)

    with pytest.raises(HTTPException) as exc:
        await avatars_router.upload_avatar(
            task_id=13,
            image=_make_upload_file(valid=True),
            current_user=SimpleNamespace(id=99),
            session=session,
        )

    assert exc.value.status_code == 403


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_avatar_invalid_image():
    task = Task(id=14, title="t", description="d", author_id=7)
    session = DummyAvatarSession(task=task)

    with pytest.raises(HTTPException) as exc:
        await avatars_router.upload_avatar(
            task_id=14,
            image=_make_upload_file(valid=False),
            current_user=SimpleNamespace(id=7),
            session=session,
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Невалидное изображение"
