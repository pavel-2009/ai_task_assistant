"""Prometheus metrics and decorators for Celery tasks."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from prometheus_client import Counter, Histogram


celery_task_duration = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration in seconds",
    ["task_name"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
)

celery_task_total = Counter(
    "celery_task_total",
    "Total number of Celery task executions",
    ["task_name", "status"],
)


def track_celery_task(task_name: str) -> Callable[..., Any]:
    """Decorator for tracking task execution time and status."""

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
