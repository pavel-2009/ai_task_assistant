"""Compatibility exports for ORM models and Pydantic schemas."""

from app.db_models import Text
from app.schemas import Recommendation, RecommendationGet

from .task import Task, TaskBase, TaskCreate, TaskGet, TaskUpdate
from .user import User, UserBase, UserCreate, UserGet

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
