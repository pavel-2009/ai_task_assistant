"""Pydantic схемы пользователей."""

from pydantic import BaseModel, Field, field_validator


class UserBase(BaseModel):
    """Базовая схема пользователя."""

    username: str = Field(min_length=1, max_length=20, description="Имя пользователя")


class UserCreate(UserBase):
    """Схема создания пользователя."""

    password: str = Field(min_length=8, max_length=128, description="Пароль пользователя (минимум 8 символов, с заглавными, строчными, цифрами и спецсимволами)")
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Проверка сложности пароля"""
        if len(v) < 8:
            raise ValueError("Пароль должен быть не менее 8 символов")
        if not any(char.isupper() for char in v):
            raise ValueError("Пароль должен содержать заглавные буквы")
        if not any(char.islower() for char in v):
            raise ValueError("Пароль должен содержать строчные буквы")
        if not any(char.isdigit() for char in v):
            raise ValueError("Пароль должен содержать цифры")
        if not any(char in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for char in v):
            raise ValueError("Пароль должен содержать специальные символы (!@#$%^&*...)")
        return v


class UserGet(UserBase):
    """Схема ответа пользователя."""

    id: int
