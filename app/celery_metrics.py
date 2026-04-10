"""Метрики Prometheus и декораторы для задач Celery."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from prometheus_client import Counter, Histogram, REGISTRY


def _get_or_create_histogram(name: str, documentation: str, labelnames: list, buckets: list = None):
    """Получить гистограмму из реестра или создать, если она отсутствует."""
    # Проверяем, зарегистрирована ли уже метрика
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, labelnames, buckets=buckets)


def _get_or_create_counter(name: str, documentation: str, labelnames: list):
    """Получить счётчик из реестра или создать, если он отсутствует."""
    # Проверяем, зарегистрирована ли уже метрика
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames)


celery_task_duration = _get_or_create_histogram(
    "celery_task_duration_seconds",
    "Celery task duration in seconds",
    ["task_name"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

celery_task_total = _get_or_create_counter(
    "celery_task_total",
    "Total number of Celery task executions",
    ["task_name", "status"],
)


def track_celery_task(task_name: str) -> Callable[..., Any]:
    """Декоратор для отслеживания времени выполнения и статуса задачи."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                celery_task_total.labels(task_name=task_name, status="success").inc()
                return result
            except Exception:
                celery_task_total.labels(task_name=task_name, status="error").inc()
                raise
            finally:
                duration = time.perf_counter() - start_time
                celery_task_duration.labels(task_name=task_name).observe(duration)

        return wrapper

    return decorator
