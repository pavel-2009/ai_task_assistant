from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest, REGISTRY


def _get_or_create_counter(name: str, documentation: str, labelnames: list):
    """Get counter from registry or create if not exists."""
    # Check if already registered
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, documentation, labelnames)


def _get_or_create_histogram(name: str, documentation: str, labelnames: list):
    """Get histogram from registry or create if not exists."""
    # Check if already registered
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, documentation, labelnames)


REQUEST_COUNT = _get_or_create_counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = _get_or_create_histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
)


def setup_metrics(app: FastAPI, enabled: bool = True, path: str = "/metrics") -> None:
    if not enabled:
        return

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)

        if request.url.path != path:
            duration = time.perf_counter() - start_time
            method = request.method
            request_path = request.url.path
            status = str(response.status_code)
            REQUEST_COUNT.labels(method=method, path=request_path, status=status).inc()
            REQUEST_LATENCY.labels(method=method, path=request_path).observe(duration)

        return response

    @app.get(path, include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
