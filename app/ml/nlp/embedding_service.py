"""
Сервис с получением эмбеддингов для текста.
"""

from __future__ import annotations

import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer

from app.ml.metrics import MLMetricsCollector


class EmbeddingService:
    """Сервис для получения эмбеддингов текста."""

    def __init__(self) -> None:
        self.metrics = MLMetricsCollector(self.__class__.__name__)
        load_start = time.perf_counter()
        self.model: SentenceTransformer = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.metrics.record_load_time(time.perf_counter() - load_start)
        self.dimension = self.model.get_sentence_embedding_dimension() # Получаем размерность эмбеддингов модели (384 для all-MiniLM-L6-v2)

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
        try:
            with self.metrics.time_inference():
                normalized_text = self._normalize_text(text)
                result = np.asarray(
                    self.model.encode(
                        normalized_text,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ),
                    dtype=np.float32,
                )
            self.metrics.record_success()
            return result
        except Exception as e:
            self.metrics.record_error(type(e).__name__)
            raise

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Получить эмбеддинги для списка текстов."""
        try:
            with self.metrics.time_inference():
                if not texts:
                    raise ValueError("Список текстов не может быть пустым")

                normalized_texts = [self._normalize_text(text) for text in texts]
                result = np.asarray(
                    self.model.encode(
                        normalized_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ),
                    dtype=np.float32,
                )
            self.metrics.record_success()
            return result
        except Exception as e:
            self.metrics.record_error(type(e).__name__)
            raise

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычислить косинусное сходство между двумя векторами."""
        if vec1.shape != vec2.shape:
            raise ValueError("Векторы должны иметь одинаковую размерность")

        return float(np.dot(vec1, vec2))
