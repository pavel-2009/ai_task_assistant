"""
Детектор дрейфа для мониторинга моделей машинного обучения.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DriftDetector:
    """Детектор дрейфа для мониторинга моделей машинного обучения."""
    
    def __init__(
        self,
        threshold: float = 0.15,
        reference_embeddings: list[np.ndarray] | None = None,
        _classifier: LogisticRegression | None = None
    ):
        self.threshold = threshold
        
        # Создаем хранилище референсных данных для сравнения
        self.reference_embeddings = reference_embeddings
        
        # Создаем внутренний классификатор
        self._classifier = _classifier if _classifier is not None else LogisticRegression()
        
        
    def set_reference(self, data: list[np.ndarray]):
        """Устанавливает референсные данные для сравнения."""
        self.reference_embeddings = data
        
        
    def calculate_drift(self, current_embedding: list[np.ndarray]) -> dict:
        """Вычисляет уровень дрейфа между текущими данными и референсными данными."""
        
        if not self.reference_embeddings:
            return {"drift_detected": False, "drift_score": 0.0, "error": "no reference"}
        
        # Создаем обучающую выборку для классификатора, объединяя референсные данные и текущие данные
        X = np.vstack([self.reference_embeddings, current_embedding])
        
        # Создаем метки классов для обучения классификатора (0 - референс, 1 - текущие данные)
        y = np.array([0] * len(self.reference_embeddings) + [1] * len(current_embedding))  # 0 - референс, 1 - текущие данные
        
        X, y = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self._classifier.fit(X, y)
        
        # Предсказываем метки классов для текущих данных
        y_pred = self._classifier.predict(current_embedding)
        
        accuracy = accuracy_score([1] * len(current_embedding), y_pred)
        
        drift_score = accuracy - 0.5  # Дрейф определяется как отклонение от случайного угадывания
        
        drift_detected = drift_score > self.threshold
        
        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "accuracy": accuracy
        }
    
    
    def add_embeddings(self, new_embeddings: list[np.ndarray]):
        """Добавляет новые эмбеддинги в референсные данные."""
        if self.reference_embeddings is None:
            self.reference_embeddings = new_embeddings
        else:
            self.reference_embeddings.extend(new_embeddings)
            
            
    def get_status(self) -> dict:
        """Возвращает текущий статус детектора дрейфа."""
        return {
            "reference_count": len(self.reference_embeddings) if self.reference_embeddings else 0,
            "threshold": self.threshold
        }
        