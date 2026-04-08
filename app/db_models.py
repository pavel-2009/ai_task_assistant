"""ORM-модели приложения на SQLAlchemy."""

from sqlalchemy import Column, Integer, String, DateTime, Enum as SqlEnum, ForeignKey, Index
from enum import Enum

from app.db import Base

from datetime import datetime


# Модель задачи для базы данных
class Task(Base):
    """Модель задачи для базы данных."""

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(20), nullable=False)
    description = Column(String(400), nullable=True)
    author_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    avatar_file = Column(String(256), nullable=True)
    tags = Column(String(256), nullable=True)
    
    # Индекс для быстрого поиска задач по автору и тегам
    __table_args__ = (
        Index("idx_author_id", "author_id"),
        Index("idx_tags", "tags"),
    )


# Модель пользователя для базы данных
class User(Base):
    """Модель пользователя."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(20), unique=True, nullable=False)
    password = Column(String(128), nullable=False)
    
    # Индекс для быстрого поиска пользователя по имени
    __table_args__ = (
        Index("idx_username", "username"),
    )


# Модель текста для NLP модуля
class Text(Base):
    """Модель текста NLP модуля."""

    __tablename__ = "texts"

    id = Column(Integer, primary_key=True, index=True)
    text_id = Column(String(64), unique=True, nullable=False)
    text = Column(String(400), nullable=False)
    
    # Индекс для быстрого поиска текста по text_id
    __table_args__ = (
        Index("idx_text_id", "text_id"),
    )
    
    
# Список возможных статусов задачи для рекомендательной системы
class Event(Enum):
    """Список возможных статусов задачи для рекомендательной системы."""
    
    VIEW = "view"
    LIKE = "like"
    CREATE = "create"
    
    
# Модель взаимодействия пользователя с задачой для рекомендательной системы
class Interaction(Base):
    """Модель взаимодействия пользователя с задачей"""
    
    __tablename__ = "interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(SqlEnum(Event), nullable=False)
    weight = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Индекс для быстрого поиска взаимодействий по пользователю и задаче
    __table_args__ = (
        # Индекс для быстрого поиска взаимодействий по пользователю и задаче
        Index("idx_user_task", "user_id", "task_id"),
    )
