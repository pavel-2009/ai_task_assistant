"""Роутер для рекомендательной системы."""

from fastapi import APIRouter, HTTPException, status, Query, Depends

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import services
from app.db_models import Task, User
from app.schemas import Recommendation, RecommendationGet
from app.db import get_async_session
from app.auth import get_current_user
from app.ml.recsys.content_based import ContentBasedRecommender
from app.ml.recsys.collaborative_filtering import CollaborativeFilteringRecommender


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
    

@router.get("/cf/recommendations", response_model=RecommendationGet, status_code=status.HTTP_200_OK)
async def get_cf_recommendations(
    current_user: User = Depends(get_current_user),
    top_k: int = Query(5, ge=1, le=20),
    session: AsyncSession = Depends(get_async_session)
) -> RecommendationGet:
    """Получение рекомендаций на основе коллаборативной фильтрации для текущего пользователя."""
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не аутентифицирован"
        )
    
    collaborative_filtering_recommender: CollaborativeFilteringRecommender = services.get_service("collaborative_filtering_recommender")
    
    recommendations = await collaborative_filtering_recommender.recommend(current_user.id, session, top_k=top_k)
    
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
    