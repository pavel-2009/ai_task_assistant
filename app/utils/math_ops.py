"""
Реализация математических операций для использования в задачах
"""

import numpy


def cosine_similarity(vec1: numpy.ndarray, vec2: numpy.ndarray) -> float:
    """Вычисление косинусного сходства между двумя векторами"""
    
    if vec1.shape != vec2.shape:
        raise ValueError("Векторы должны иметь одинаковую размерность")
    
    dot_product = numpy.dot(vec1, vec2)
    norm_vec1 = numpy.linalg.norm(vec1)
    norm_vec2 = numpy.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)