"""
Pydantic модели для валидации
"""

from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String

from app.db import Base


# === Pydantic модели ===
class TaskBase(BaseModel):
    """Базовый класс задачи"""
    title: str = Field(min_length=1, max_length=20, description="Название задачи")
    description: str = Field(max_length=400, description="Описание задачи")



class TaskCreate(TaskBase):
    """Класс для создания задачи"""
    pass


class TaskUpdate(TaskBase):
    """Класс для обновления задачи"""
    pass


class TaskGet(TaskBase):
    """Класс для получения задачи"""
    id: int
    author_id: int = Field(description="ID автора задачи")


class UserBase(BaseModel):
    """Базовый класс пользователя"""
    username: str = Field(min_length=1, max_length=20, description="Имя пользователя")



class UserCreate(UserBase):
    """Класс для создания пользователя"""
    password: str = Field(min_length=6, max_length=128, description="Пароль пользователя")


class UserGet(UserBase):
    """Класс для получения пользователя"""
    id: int


# === SQLAlchemy модели ===
class Task(Base):
    """Модель задачи для базы данных"""
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(20), nullable=False)
    description = Column(String(400), nullable=True)
    author_id = Column(Integer, nullable=False)


class User(Base):
    """Модель пользователя"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(20), unique=True, nullable=False)
    password = Column(String(128), nullable=False)