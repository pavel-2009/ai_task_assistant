"""
Сервис для обработки взаимодействий пользователей с задачами и обновления рекомендаций на основе этих взаимодействий. 
"""

import redis
from scipy.sparse import csr_matrix

import numpy as np
import pickle

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core import config
from app.db_models import Interaction

import logging


logger = logging.getLogger(__name__)


class CollaborativeFilteringRecommender:
    """Сервис для обработки взаимодействий пользователей с задачами и обновления рекомендаций на основе этих взаимодействий."""
    
    def __init__(
        self,
        redis_client: redis.Redis = None,
    ):
        self.redis_client = redis_client
        
    
    async def build_user_item_matrix(self, session: AsyncSession) -> tuple[csr_matrix, dict, dict, list, list]:
        """Построение разреженной матрицы взаимодействий пользователей с задачами."""
        
        result = await session.execute(select(Interaction))
        interactions = result.scalars().all()
        
        unique_users = sorted(set(i.user_id for i in interactions))
        unique_tasks = sorted(set(i.task_id for i in interactions))
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        task_to_idx = {task: idx for idx, task in enumerate(unique_tasks)}
        idx_to_task = {idx: task for task, idx in task_to_idx.items()}
        
        rows = [user_to_idx[i.user_id] for i in interactions]
        cols = [task_to_idx[i.task_id] for i in interactions]
        data = [i.weight for i in interactions]
        
        matrix = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_tasks)))
        
        return matrix, user_to_idx, idx_to_task, unique_users, unique_tasks
        
        
    def load(self) -> tuple[np.ndarray, np.ndarray, dict, dict, dict, dict, list] | tuple[None, None, None, None, None, None, list]:
        """Загрузка модели из Redis."""
        
        model_data = self.redis_client.get("collaborative_filtering_model")
        if model_data:
            user_factors, task_factors, user_to_idx, idx_to_task, unique_users, unique_tasks, popular_tasks = pickle.loads(model_data)  # Десериализация модели из Redis
            
            if user_factors.shape[0] != len(unique_users) or task_factors.shape[0] != len(unique_tasks):
                logger.error("Размерности факторов не совпадают с количеством уникальных пользователей или задач при загрузке модели")
                return None, None, None, None, None, None, []
            return user_factors, task_factors, user_to_idx, idx_to_task, unique_users, unique_tasks, popular_tasks
        return None, None, None, None, None, None, []


    def recommend(self, user_id: int, top_k: int = config.DEFAULT_TOP_K) -> list[tuple[int, float]]:
        """Получение рекомендаций для пользователя на основе обученной модели."""
        
        # Проверяем кэш
        recommendations_cache_key = f"cf_recommendations_user_{user_id}"
        cached_recommendations = self.redis_client.get(recommendations_cache_key)
        if cached_recommendations:
            return pickle.loads(cached_recommendations)  # Десериализация рекомендаций из кэша
        
        user_factors, task_factors, user_to_idx, idx_to_task, unique_users, unique_tasks, popular_tasks = self.load()
        if user_factors is None:
            return [(task_id, 0.0) for task_id in popular_tasks]  # Рекомендации на основе популярных задач для новых пользователей (холодный старт)
        
        if user_factors.shape[0] != len(unique_users) or task_factors.shape[0] != len(unique_tasks):
            logger.error("Размерности факторов не совпадают с количеством уникальных пользователей или задач при загрузке модели")
            return [(task_id, 0.0) for task_id in popular_tasks]  # Рекомендации на основе популярных задач для новых пользователей (холодный старт)
        
        # Если пользователь не найден в модели, возвращаем рекомендации на основе популярных задач (холодный старт)
        if user_id not in user_to_idx:
            return [(task_id, 0.0) for task_id in popular_tasks]  # Рекомендации на основе популярных задач для новых пользователей (холодный старт)
        
        # Получаем вектор факторов для данного пользователя
        user_idx = user_to_idx[user_id]
        user_vector = user_factors[user_idx]
        
        # Вычисляем предсказанные оценки для всех объектов
        scores = task_factors.dot(user_vector)
        
        # Получаем топ-K рекомендаций
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = [(idx_to_task[idx], scores[idx]) for idx in top_k_indices]
        
        self.redis_client.set(recommendations_cache_key, pickle.dumps(recommendations), ex=3600)  # Кэшируем рекомендации на 1 час
        
        return recommendations
