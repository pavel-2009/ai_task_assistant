"""Pydantic схемы рекомендаций."""

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    """Схема отдельной рекомендации."""

    task_id: int = Field(description="ID задачи")
    description: str = Field(description="Описание задачи")
    similarity_score: float = Field(description="Оценка похожести задачи")


class RecommendationGet(BaseModel):
    """Схема ответа рекомендательной системы."""

    recommendations: list[Recommendation] = Field(description="Список рекомендованных задач")
