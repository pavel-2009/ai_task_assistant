"""
Сервис с получением эмбеддингов для текста.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Сервис для получения эмбеддингов текста."""

    def __init__(self) -> None:
        self.model: SentenceTransformer = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.dimension = self.model.get_sentence_embedding_dimension()

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Текст должен быть строкой")

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Текст не может быть пустым")

        return normalized_text

    def encode_one(self, text: str) -> np.ndarray:
        """Получить эмбеддинг для одного текста."""
        normalized_text = self._normalize_text(text)
        return np.asarray(
            self.model.encode(
                normalized_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Получить эмбеддинги для списка текстов."""
        if not texts:
            raise ValueError("Список текстов не может быть пустым")

        normalized_texts = [self._normalize_text(text) for text in texts]
        return np.asarray(
            self.model.encode(
                normalized_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычислить косинусное сходство между двумя векторами."""
        if vec1.shape != vec2.shape:
            raise ValueError("Векторы должны иметь одинаковую размерность")

        return float(np.dot(vec1, vec2))
