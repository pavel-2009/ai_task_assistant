from datetime import timedelta
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi import HTTPException

from app.auth import create_access_token, get_current_user
from app.core.config import config


def test_create_access_token():
    token = create_access_token(user_id=123)
    payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    assert payload['user_id'] == 123


def test_create_access_token_custom_expires():
    token = create_access_token(user_id=123, expires_delta=timedelta(minutes=30))
    payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    assert payload['user_id'] == 123
    assert 'exp' in payload


@pytest.mark.asyncio
async def test_invalid_token_handling():
    session = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await get_current_user(token='invalid.token.here', session=session)

    assert exc.value.status_code == 401
