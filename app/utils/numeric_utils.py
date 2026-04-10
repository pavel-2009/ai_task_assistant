"""Числовые утилиты для ML-кода и рекомендательной системы."""

import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Вычислить косинусное сходство для двух векторов одинаковой формы."""

    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape")

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return float(dot_product / (norm_vec1 * norm_vec2))
