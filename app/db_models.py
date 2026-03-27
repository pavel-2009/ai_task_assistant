"""SQLAlchemy ORM модели приложения."""

from sqlalchemy import Column, Integer, String

from app.db import Base


class Task(Base):
    """Модель задачи для базы данных."""

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(20), nullable=False)
    description = Column(String(400), nullable=True)
    author_id = Column(Integer, nullable=False)
    avatar_file = Column(String(256), nullable=True)
    tags = Column(String(256), nullable=True)


class User(Base):
    """Модель пользователя."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(20), unique=True, nullable=False)
    password = Column(String(128), nullable=False)


class Text(Base):
    """Модель текста NLP модуля."""

    __tablename__ = "texts"

    id = Column(Integer, primary_key=True, index=True)
    text_id = Column(String(64), unique=True, nullable=False)
    text = Column(String(400), nullable=False)
