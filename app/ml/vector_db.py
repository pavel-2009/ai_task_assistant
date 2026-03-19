"""
Модуль для работы с векторной базой данных. Здесь будет класс VectorDB, который будет хранить эмбеддинги и обеспечивать поиск по ним. 
"""

import faiss
import numpy as np


class VectorDB:
    """Класс для хранения эмбеддингов и поиска по ним"""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []  # Список текстов, соответствующих эмбеддингам в индексе
    
    
    def add(self, embedding: list[float], text: str):
        """Добавить эмбеддинг и связанный с ним текст в базу данных"""
        self.index.add(np.array([embedding], dtype=np.float32))
        self.texts.append(text)

        
        
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """Поиск наиболее похожих текстов по эмбеддингу запроса"""
        
        if not query_embedding:
            raise ValueError("Эмбеддинг запроса не может быть пустым")
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue  # Нет больше результатов
            results.append((idx, dist, self.texts[idx]))

        return results