"""ML-specific Prometheus metrics utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Histogram, REGISTRY


# Глобальные метрики ML (создаются один раз)
_ml_inference_duration = None
_ml_inference_requests = None
_ml_model_load_duration = None


def _init_ml_metrics():
    """Инициализирует глобальные ML метрики один раз."""
    global _ml_inference_duration, _ml_inference_requests, _ml_model_load_duration
    
    if _ml_inference_duration is not None:
        return  # Уже инициализированы
    
    try:
        _ml_inference_duration = Histogram(
            "ml_inference_duration_seconds",
            "ML inference duration in seconds",
            ["model"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        )
    except ValueError:
        # Уже существует в реестре, получаем его
        for collector in REGISTRY._collector_to_names:
            if getattr(collector, '_name', None) == "ml_inference_duration_seconds":
                _ml_inference_duration = collector
                break
    
    try:
        _ml_inference_requests = Counter(
            "ml_inference_requests_total",
            "Total ML inference requests",
            ["model", "status", "error_type"],
        )
    except ValueError:
        for collector in REGISTRY._collector_to_names:
            if getattr(collector, '_name', None) == "ml_inference_requests_total":
                _ml_inference_requests = collector
                break
    
    try:
        _ml_model_load_duration = Histogram(
            "ml_model_load_duration_seconds",
            "ML model loading duration in seconds",
            ["model"],
        )
    except ValueError:
        for collector in REGISTRY._collector_to_names:
            if getattr(collector, '_name', None) == "ml_model_load_duration_seconds":
                _ml_model_load_duration = collector
                break


class MLMetricsCollector:
    """Сборщик метрик для ML-сервисов."""

    def __init__(self, model_name: str) -> None:
        _init_ml_metrics()
        self.model_name = model_name
        self.inference_duration = _ml_inference_duration
        self.inference_requests = _ml_inference_requests
        self.model_load_duration = _ml_model_load_duration

    @contextmanager
    def time_inference(self) -> Iterator[None]:
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.inference_duration.labels(model=self.model_name).observe(duration)

    def record_success(self) -> None:
        self.inference_requests.labels(
            model=self.model_name,
            status="success",
            error_type="none",
        ).inc()

    def record_error(self, error_type: str) -> None:
        self.inference_requests.labels(
            model=self.model_name,
            status="error",
            error_type=error_type,
        ).inc()

    def record_load_time(self, seconds: float) -> None:
        self.model_load_duration.labels(model=self.model_name).observe(seconds)
