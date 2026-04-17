"""Юнит-тесты для auth-утилит без API-клиента."""

from datetime import timedelta

import pytest
from fastapi import HTTPException

from app import auth


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _DummySession:
    def __init__(self, user=None):
        self.user = user

    async def execute(self, _query):
        return _ScalarResult(self.user)


@pytest.mark.unit
def test_create_access_token_contains_user_id():
    token = auth.create_access_token(user_id=123, expires_delta=timedelta(minutes=5))
    payload = auth.jwt.decode(token, auth.config.SECRET_KEY, algorithms=[auth.config.JWT_ALGORITHM])
    assert payload["user_id"] == 123
    assert "exp" in payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_user_invalid_token():
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(token="broken.token.value", session=_DummySession())

    assert exc.value.status_code == 401


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_current_user_user_not_found():
    token = auth.create_access_token(user_id=999)

    with pytest.raises(HTTPException) as exc:
        await auth.get_current_user(token=token, session=_DummySession(user=None))

    assert exc.value.status_code == 401
    assert "User not found" in exc.value.detail
