"""
Модуль для аутентификации пользователей.
"""

import jwt
from jwt.exceptions import InvalidTokenError
import bcrypt
from datetime import datetime, timedelta
from fastapi import HTTPException
from sqlalchemy import select

import os
from dotenv import load_dotenv

from app.models import User
from app.db import get_async_session

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")


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
