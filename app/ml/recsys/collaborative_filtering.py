"""
Сервис для обработки взаимодействий пользователей с задачами и обновления рекомендаций на основе этих взаимодействий. 
"""

import redis
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

import numpy as np
import pickle

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db_models import Interaction, Event


class CollaborativeFilteringRecommender:
    """Сервис для обработки взаимодействий пользователей с задачами и обновления рекомендаций на основе этих взаимодействий."""
    
    def __init__(
        self,
        redis_client: redis.Redis = None,
    ):
        self.redis_client = redis_client
        
    
    async def build_user_item_matrix(self, session: AsyncSession) -> csr_matrix:
        """Построение разреженной матрицы пользователь-объект на основе взаимодействий из БД."""
        
        interactions = await session.execute(
            select(Interaction)
        )
        interactions = interactions.scalars().all()
        
        user_ids_to_tasks = [(interaction.user_id, interaction.task_id) for interaction in interactions]
        
        # Создаем разреженную матрицу пользователь-объект
        user_item_matrix = csr_matrix(
            (interaction.weight for interaction in interactions),
            (
                [user_id for user_id, _ in user_ids_to_tasks],
                [task_id for _, task_id in user_ids_to_tasks]
            )
        )
        return user_item_matrix
    
    
    def train_model(self, user_item_matrix: csr_matrix):
        """Обучение модели коллаборативной фильтрации на основе разреженной матрицы."""
        
        model: AlternatingLeastSquares = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
        
        model.fit(user_item_matrix)
        
        
    def load(self) -> AlternatingLeastSquares:
        """Загрузка модели из Redis."""
        
        model_data = self.redis_client.get("collaborative_filtering_model")
        if model_data:
            model = np.loads(model_data)  # Десериализация модели из Redis
            return model
        return None
    
    
    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Получение рекомендаций для пользователя на основе обученной модели."""
        
        model = self.load()
        if model is None:
            return []
        
        user_factors, item_factors = model
        
        # Получаем вектор факторов для данного пользователя
        user_vector = user_factors[user_id]
        
        # Вычисляем предсказанные оценки для всех объектов
        scores = item_factors.dot(user_vector)
        
        # Получаем топ-K рекомендаций
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(index, scores[index]) for index in top_k_indices]