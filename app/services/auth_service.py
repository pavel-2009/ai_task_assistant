"""Бизнес-логика аутентификации, изолированная от HTTP-роутеров."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import create_access_token
from app.core.security import hash_password, verify_password
from app.db_models import User
from app.schemas import TokenResponse, UserCreate, UserGet


class AuthService:
    """Операции регистрации и входа пользователя."""

    async def register_user(self, session: AsyncSession, user_payload: UserCreate) -> UserGet:
        if not user_payload.username or not user_payload.password:
            raise HTTPException(status_code=400, detail="Username and password must not be empty")

        existing_user = await session.execute(select(User).where(User.username == user_payload.username))
        if existing_user.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="User with this username already exists")

        new_user = User(username=user_payload.username, password=hash_password(user_payload.password))
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        return UserGet(id=new_user.id, username=new_user.username)

    async def login_user(
        self,
        session: AsyncSession,
        username: str,
        password: str,
    ) -> TokenResponse:
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password must not be empty")

        existing_user = await session.execute(select(User).where(User.username == username))
        user = existing_user.scalar_one_or_none()
        if user is None or not verify_password(password, user.password):
            raise HTTPException(status_code=400, detail="Invalid username or password")

        return TokenResponse(access_token=create_access_token(user_id=user.id), token_type="bearer")
