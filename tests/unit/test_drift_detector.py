import numpy as np

from app.ml.monitoring.drift_detector import DriftDetector


def test_without_reference_data():
    detector = DriftDetector()
    result = detector.calculate_drift([np.array([0.1, 0.2])])
    assert result['drift_detected'] is False
    assert result['drift_score'] == 0.0
    assert result['error'] == 'no reference'


def test_add_one_embedding():
    detector = DriftDetector()
    detector.add_embedding(np.array([1.0, 2.0]))
    assert detector.get_status()['reference_count'] == 1


def test_add_multiple_embeddings():
    detector = DriftDetector()
    detector.add_embedding(np.array([1.0, 2.0]))
    detector.add_embedding(np.array([2.0, 3.0]))
    detector.add_embedding(np.array([3.0, 4.0]))
    assert detector.get_status()['reference_count'] == 3


def test_get_status_empty():
    detector = DriftDetector(threshold=0.12)
    status = detector.get_status()
    assert status == {'reference_count': 0, 'threshold': 0.12}


def test_get_status_with_data():
    detector = DriftDetector(threshold=0.2)
    detector.set_reference([np.array([0.0, 0.0])])
    status = detector.get_status()
    assert status == {'reference_count': 1, 'threshold': 0.2}


def test_similar_data_small_drift():
    rng = np.random.default_rng(42)
    ref = [rng.normal(0, 1, size=4) for _ in range(40)]
    current = [rng.normal(0, 1, size=4) for _ in range(40)]
    detector = DriftDetector(threshold=0.2, reference_embeddings=ref)
    result = detector.calculate_drift(current)
    assert result['drift_detected'] is False


def test_different_data_big_drift():
    rng = np.random.default_rng(42)
    ref = [rng.normal(0, 1, size=4) for _ in range(40)]
    current = [rng.normal(5, 1, size=4) for _ in range(40)]
    detector = DriftDetector(threshold=0.2, reference_embeddings=ref)
    result = detector.calculate_drift(current)
    assert result['drift_detected'] is True
