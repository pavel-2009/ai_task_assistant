"""
Модуль для работы с векторной базой данных. Здесь будет класс VectorDB, который будет хранить эмбеддинги и обеспечивать поиск по ним. 
"""

import faiss
import numpy as np
import pickle
import redis.asyncio as redis
import io


class VectorDB:
    """Класс для хранения эмбеддингов и поиска по ним с поддержкой Redis"""
    
    def __init__(self, dim: int, redis_client: redis.Redis = None):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []  # Список текстов, соответствующих эмбеддингам в индексе
        self.redis_client = redis_client
        self.index_key = "vector_db:faiss_index"
        self.texts_key = "vector_db:texts"
    
    
    def add(self, embedding: list[float], text: str):
        """Добавить эмбеддинг и связанный с ним текст в базу данных"""
        self.index.add(np.array([embedding], dtype=np.float32))
        self.texts.append(text)

        
        
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Поиск наиболее похожих текстов по эмбеддингу запроса"""
        
        if not query_embedding:
            raise ValueError("Эмбеддинг запроса не может быть пустым")
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue  # Нет больше результатов
            # Конвертируем L2 distance в similarity (меньше расстояние = выше similarity)
            similarity = float(1.0 / (1.0 + float(dist)))  # Пример конверсии, можно использовать другую формулу
            results.append({
                "index": int(idx),
                "text": self.texts[idx],
                "similarity": float(similarity)
            })

        return results
    
    
    def save_to_redis(self) -> bool:
        """Сохранить индекс и тексты в Redis"""
        if not self.redis_client:
            return False
        try:
            # Сохраняем FAISS индекс
            index_bytes = io.BytesIO()
            faiss.write_index(self.index, index_bytes)
            self.redis_client.set(self.index_key, index_bytes.getvalue())
            
            # Сохраняем список текстов
            texts_bytes = pickle.dumps(self.texts)
            self.redis_client.set(self.texts_key, texts_bytes)
            
            return True
        except Exception as e:
            print(f"Ошибка при сохранении индекса в Redis: {e}")
            return False
    
    
    def load_from_redis(self) -> bool:
        """Загрузить индекс и тексты из Redis"""
        if not self.redis_client:
            return False
        try:
            # Загружаем FAISS индекс
            index_bytes = self.redis_client.get(self.index_key)
            if index_bytes:
                index_io = io.BytesIO(index_bytes)
                self.index = faiss.read_index(index_io)
            
            # Загружаем список текстов
            texts_bytes = self.redis_client.get(self.texts_key)
            if texts_bytes:
                self.texts = pickle.loads(texts_bytes)
            
            return True
        except Exception as e:
            print(f"Ошибка при загрузке индекса из Redis: {e}")
            return False