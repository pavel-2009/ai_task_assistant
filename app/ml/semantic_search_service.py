"""
Сервис для семантического поиска по эмбеддингам. Использует модель SentenceTransformer для получения эмбеддингов текстов и вычисления сходства между ними.
"""

from .embedding_service import EmbeddingService
from .vector_db import VectorDB


class SemanticSearchService:
    """Сервис для семантического поиска по эмбеддингам"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service: EmbeddingService = embedding_service
        self.database = VectorDB(dim=384) 
        
        
    def index(self, text: str):
        """Индексировать текст, добавляя его эмбеддинг в базу данных"""
        
        if not text or len(text) == 0:
            raise ValueError("Текст для индексирования не может быть пустым")
        
        embedding = self.embedding_service.encode_one(text)
        self.database.add(embedding, text)
    
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Искать документы, наиболее похожие на запрос"""
        
        if not query or len(query) == 0:
            raise ValueError("Запрос не может быть пустым")
        
        query_embedding = self.embedding_service.encode_one(query)
        
        if not self.database:
            raise ValueError("База данных документов пуста. Индексируйте документы перед поиском.")
        
        results = self.database.search(query_embedding, top_k=top_k)
        
        # Сортируем документы по сходству и возвращаем топ K
        sorted_docs = sorted(results, key=lambda x: x["similarity"], reverse=True)
        return sorted_docs[:top_k]