"""Тесты для проверки работы детектора дрейфа данных."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_drift_detector_no_reference():
    """Проверяет, что при отсутствии референсных данных возвращается правильный статус."""
    from app.ml.monitoring.drift_detector import DriftDetector
    
    drift_detector = DriftDetector()
    
    current_embedding = [1.0, 2.0, 3.0]
    
    result = drift_detector.calculate_drift(current_embedding)
    
    assert result["drift_detected"] is False
    assert result["drift_score"] == 0.0
    assert result["error"] == "no reference"
    
    
def test_drift_detector_same_distribution():
    """Проверяет, что при одинаковом распределении данных дрейф не обнаруживается."""
    from app.ml.monitoring.drift_detector import DriftDetector
    import numpy as np
    
    reference_embeddings = [np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1])]
    current_embedding = [np.array([1.05, 2.05, 3.05])]
    
    drift_detector = DriftDetector(threshold=0.1)
    drift_detector.set_reference(reference_embeddings)
    
    result = drift_detector.calculate_drift(current_embedding)
    
    assert result["drift_detected"] is False
    assert result["drift_score"] < 0.1
    
    
def test_drift_detector_different_distribution():
    """Проверяет, что при разных распределениях данных дрейф обнаруживается."""
    from app.ml.monitoring.drift_detector import DriftDetector
    import numpy as np
    
    reference_embeddings = [np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1])]
    current_embedding = [np.array([4.0, 5.0, 6.0])]
    
    drift_detector = DriftDetector(threshold=0.1)
    drift_detector.set_reference(reference_embeddings)
    
    result = drift_detector.calculate_drift(current_embedding)
    
    assert result["drift_detected"] is True
    assert result["drift_score"] > 0.1
    

def test_monitoring_endpoints():
    """Проверяет, что эндпоинты мониторинга возвращают корректный статус."""
    from app.ml.monitoring.drift_detector import DriftDetector
    
    # Тест без инициализации полного приложения (чтобы избежать Redis подключения)
    drift_detector = DriftDetector(threshold=0.15)
    
    # Тест GET /monitoring/drift/report  
    status = drift_detector.get_status()
    assert "reference_count" in status
    assert "threshold" in status
    assert status["threshold"] == 0.15
    
    # Тест добавления данных
    import numpy as np
    reference_data = [np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1])]
    drift_detector.set_reference(reference_data)
    
    status = drift_detector.get_status()
    assert status["reference_count"] == 2
