"""Pydantic схемы пользователей."""

from pydantic import BaseModel, Field


class UserBase(BaseModel):
    """Базовая схема пользователя."""

    username: str = Field(min_length=1, max_length=20, description="Имя пользователя")


class UserCreate(UserBase):
    """Схема создания пользователя."""

    password: str = Field(min_length=6, max_length=128, description="Пароль пользователя")


class UserGet(UserBase):
    """Схема ответа пользователя."""

    id: int
