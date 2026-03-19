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
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.database_service = None  # Здесь будет ссылка на сервис для работы с базой данных, если он понадобится в будущем

    def encode_one(self, text: str) -> np.ndarray:
        """Получить эмбеддинг для одного текста"""
        if text is None or len(text) == 0:
            raise ValueError("Текст не может быть пустым")

        return np.asarray(
            self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True),
            dtype=np.float32,
        )

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Получить эмбеддинги для списка текстов"""
        if len(texts) == 0:
            raise ValueError("Список текстов не может быть пустым")

        if not all(isinstance(t, str) for t in texts):
            raise ValueError("Все элементы в списке должны быть строками")

        if any(len(t) == 0 for t in texts):
            raise ValueError("Тексты не могут быть пустыми")

        return np.asarray(
            self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
            dtype=np.float32,
        )

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычислить косинусное сходство между двумя векторами"""
        vec1 = np.asarray(vec1, dtype=np.float32)
        vec2 = np.asarray(vec2, dtype=np.float32)

        if vec1.shape != vec2.shape:
            raise ValueError("Векторы должны иметь одинаковую размерность")

        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
