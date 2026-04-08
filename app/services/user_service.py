"""Утилиты бизнес-логики, связанные с пользователями."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_models import User


class UserService:
    """Лёгкий сервис поиска пользователей, вынесенный из роутеров."""

    async def get_by_username(self, session: AsyncSession, username: str) -> User | None:
        result = await session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()
