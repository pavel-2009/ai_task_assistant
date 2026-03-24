"""
Модуль для аутентификации пользователей.
"""

import jwt
from jwt.exceptions import InvalidTokenError
import bcrypt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import os
from dotenv import load_dotenv

from app.models import User
from app.db import get_async_session

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    """Хэширование пароля с bcrypt"""

    salt = bcrypt.gensalt()

    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, hashed_password: str) -> bool:
    """Проверка пароля"""

    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


def create_access_token(user_id: int, expires_delta: timedelta = timedelta(minutes=30)) -> str:
    """Создание JWT токена для пользователя"""

    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + expires_delta
    }

    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


async def get_current_user(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_async_session)) -> User:
    """Получение текущего пользователя по JWT токену"""

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("user_id")

        if user_id is None:
            raise InvalidTokenError("Invalid token")

        user = await session.execute(select(User).where(User.id == user_id))
        user = user.scalar_one_or_none()

        if user is None:
            raise InvalidTokenError("User not found")

        return user

    except InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))