"""Роутер для рекомендательной системы."""

from fastapi import APIRouter, HTTPException, status, Query, Depends

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import services
from app.models import RecommendationGet, Recommendation, Task, User
from app.db import get_async_session
from app.auth import get_current_user
from app.ml.recsys.content_based import ContentBasedRecommender


router = APIRouter(
    prefix="/recsys",
    tags=["recsys"],
)


@router.get("/recommendations/{task_id}", response_model=RecommendationGet, status_code=status.HTTP_200_OK)
async def get_recommendations(
    task_id: int,
    current_user: User = Depends(get_current_user),
    top_k: int = Query(5, ge=1, le=20),
    author_id: int = Query(None, description="ID автора задачи для фильтрации рекомендаций"),
    session: AsyncSession = Depends(get_async_session)
) -> RecommendationGet:
    """Получение рекомендаций для задачи на основе ее ID."""
    
    task = await session.execute(
        select(Task).where(Task.id == task_id)
    )
    
    task = task.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Задача не найдена"
        )
        
    content_based_recommender: ContentBasedRecommender = services.get_service("content_based_recommender")
    
    recommendations = await content_based_recommender.recommend(task.id, session, top_k=top_k, author_id=author_id or current_user.id)
    
    return RecommendationGet(
        recommendations=[
            Recommendation(
                task_id=rec["task_id"],
                description=rec["description"],
                similarity_score=rec["similarity_score"]
            )
            for rec in recommendations[:top_k]  # Ограничиваем количество рекомендаций до top_k
        ]
    )
    