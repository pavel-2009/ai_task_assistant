"""
Сервис с получением эмбеддингов для текста
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import torch


class EmbeddingService:
    """Сервис для получения эмбеддингов текста"""
    
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        
    def encode(self, text: str | list[str]) -> np.ndarray:
        """Получить эмбеддинг для текста или списка текстов"""
        batches = self.model.encode(
            [text] if isinstance(text, str) else text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return batches
