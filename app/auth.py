"""
Модуль для аутентификации пользователей.
"""

from datetime import datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config
from app.db import get_async_session
from app.db_models import User


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(user_id: int, expires_delta: timedelta | None = None) -> str:
    """Создание JWT токена для пользователя"""

    token_ttl = expires_delta or timedelta(minutes=config.JWT_EXPIRE_MINUTES)
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + token_ttl,
    }

    return jwt.encode(payload, config.SECRET_KEY, algorithm=config.JWT_ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_session),
) -> User:
    """Получение текущего пользователя по JWT токену"""

    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
        user_id = payload.get("user_id")

        if user_id is None:
            raise InvalidTokenError("Invalid token")

        user = await session.execute(select(User).where(User.id == user_id))
        user = user.scalar_one_or_none()

        if user is None:
            raise InvalidTokenError("User not found")

        return user

    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
