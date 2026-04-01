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
        # Правильно конкатенируем два списка эмбеддингов перед vstack
        X = np.vstack(self.reference_embeddings + current_embedding)
        
        # Создаем метки классов для обучения классификатора (0 - референс, 1 - текущие данные)
        y = np.array([0] * len(self.reference_embeddings) + [1] * len(current_embedding))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self._classifier.fit(X_train, y_train)
        
        # Предсказываем метки классов для тестового набора
        y_pred = self._classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        drift_score = accuracy - 0.5  # Дрейф определяется как отклонение от случайного угадывания
        
        drift_detected = drift_score > self.threshold
        
        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "accuracy": accuracy
        }
    
    
    def add_embedding(self, new_embedding: np.ndarray):
        """Добавляет новый эмбеддинг в референсные данные."""
        if self.reference_embeddings is None:
            self.reference_embeddings = [new_embedding]
        else:
            self.reference_embeddings.append(new_embedding)

    def get_status(self) -> dict:
        """Возвращает текущий статус детектора дрейфа."""
        return {
            "reference_count": len(self.reference_embeddings) if self.reference_embeddings else 0,
            "threshold": self.threshold
        }
        