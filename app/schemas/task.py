"""Pydantic схемы задач."""

from pydantic import BaseModel, ConfigDict, Field


class TaskBase(BaseModel):
    """Базовая схема задачи."""

    title: str = Field(min_length=1, max_length=20, description="Название задачи")
    description: str = Field(max_length=400, description="Описание задачи")
    avatar_file: str | None = Field(
        default=None,
        max_length=256,
        description="Путь к файлу аватара задачи",
    )
    tags: str | None = Field(default=None, max_length=256, description="Теги задачи")


class TaskCreate(TaskBase):
    """Схема создания задачи."""


class TaskUpdate(BaseModel):
    """Схема частичного обновления задачи."""

    title: str | None = Field(default=None, min_length=1, max_length=20)
    description: str | None = Field(default=None, max_length=400)
    avatar_file: str | None = Field(default=None, max_length=256)
    tags: str | None = Field(default=None, max_length=256)


class TaskGet(TaskBase):
    """Схема ответа задачи."""

    id: int
    author_id: int = Field(description="ID автора задачи")

    model_config = ConfigDict(from_attributes=True)
