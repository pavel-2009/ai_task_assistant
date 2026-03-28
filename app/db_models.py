"""SQLAlchemy ORM модели приложения."""

from sqlalchemy import Column, Integer, String, DateTime, Enum as SqlEnum
from enum import Enum

from app.db import Base


# Модель задачи для базы данных
class Task(Base):
    """Модель задачи для базы данных."""

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(20), nullable=False)
    description = Column(String(400), nullable=True)
    author_id = Column(Integer, nullable=False)
    avatar_file = Column(String(256), nullable=True)
    tags = Column(String(256), nullable=True)


# Модель пользователя для базы данных
class User(Base):
    """Модель пользователя."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(20), unique=True, nullable=False)
    password = Column(String(128), nullable=False)


# Модель текста для NLP модуля
class Text(Base):
    """Модель текста NLP модуля."""

    __tablename__ = "texts"

    id = Column(Integer, primary_key=True, index=True)
    text_id = Column(String(64), unique=True, nullable=False)
    text = Column(String(400), nullable=False)
    
    
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
    user_id = Column(Integer, nullable=False)
    task_id = Column(Integer, nullable=False)
    event_type = Column(SqlEnum(Event), nullable=False)
    weight = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
