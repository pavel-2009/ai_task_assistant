"""Совместимость импорта: ORM модели + Pydantic схемы."""

from app.db_models import Task, Text, User
from app.schemas import (
    Recommendation,
    RecommendationGet,
    TaskBase,
    TaskCreate,
    TaskGet,
    TaskUpdate,
    UserBase,
    UserCreate,
    UserGet,
)

__all__ = [
    "Recommendation",
    "RecommendationGet",
    "Task",
    "TaskBase",
    "TaskCreate",
    "TaskGet",
    "TaskUpdate",
    "Text",
    "User",
    "UserBase",
    "UserCreate",
    "UserGet",
]
