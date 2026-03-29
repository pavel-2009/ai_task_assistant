"""
Сервис для обработки взаимодействий пользователей с задачами и обновления рекомендаций на основе этих взаимодействий. 
"""

import redis
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

import numpy as np
import pickle

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db_models import Interaction, Event


class CollaborativeFilteringRecommender:
    """Сервис для обработки взаимодействий пользователей с задачами и обновления рекомендаций на основе этих взаимодействий."""
    
    def __init__(
        self,
        redis_client: redis.Redis = None,
    ):
        self.redis_client = redis_client
        
    
    def build_user_item_matrix(self, session: Session) -> tuple[csr_matrix, dict, dict, list, list]:
        """Построение разреженной матрицы взаимодействий пользователей с задачами."""
        
        interactions = session.execute(select(Interaction))
        interactions = interactions.scalars().all()
        
        unique_users = sorted(set(i.user_id for i in interactions))
        unique_tasks = sorted(set(i.task_id for i in interactions))
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        task_to_idx = {task: idx for idx, task in enumerate(unique_tasks)}
        
        rows = [user_to_idx[i.user_id] for i in interactions]
        cols = [task_to_idx[i.task_id] for i in interactions]
        data = [i.weight for i in interactions]
        
        matrix = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_tasks)))
        
        return matrix, user_to_idx, task_to_idx, unique_users, unique_tasks
        
        
    def load(self) -> AlternatingLeastSquares:
        """Загрузка модели из Redis."""
        
        model_data = self.redis_client.get("collaborative_filtering_model")
        if model_data:
            model = pickle.loads(model_data)  # Десериализация модели из Redis
            return model
        return None
    
    
    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Получение рекомендаций для пользователя на основе обученной модели."""
        
        model = self.load()
        if model is None:
            return []
        
        user_factors, item_factors = model  # Распакованные факторы из модели
        
        # Получаем вектор факторов для данного пользователя
        user_vector = user_factors[user_id]
        
        # Вычисляем предсказанные оценки для всех объектов
        scores = item_factors.dot(user_vector)
        
        # Получаем топ-K рекомендаций
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(index, scores[index]) for index in top_k_indices]