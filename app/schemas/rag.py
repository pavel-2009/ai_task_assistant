"""Схемы запросов/ответов для RAG роутера."""

from pydantic import BaseModel, Field

from app.core import config


class AskRequest(BaseModel):
    """Тело запроса к RAG endpoints."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=10)
    use_cache: bool = True
