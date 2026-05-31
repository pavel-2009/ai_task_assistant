from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.metrics import _get_or_create_counter, _get_or_create_histogram, setup_metrics


def test_create_counter():
    counter = _get_or_create_counter('test_counter_total', 'Test counter', ['label'])
    assert counter is not None


def test_create_histogram():
    histogram = _get_or_create_histogram('test_histogram_seconds', 'Test histogram', ['label'])
    assert histogram is not None


def test_get_existing_metric_from_registry():
    first = _get_or_create_counter('test_existing_counter_total', 'Existing counter', ['a'])
    second = _get_or_create_counter('test_existing_counter_total', 'Existing counter', ['a'])
    assert first is second


def test_setup_metrics_middleware_enabled():
    app = FastAPI()
    setup_metrics(app, enabled=True, path='/metrics')

    @app.get('/ping')
    async def ping():
        return {'ok': True}

    with TestClient(app) as client:
        ping = client.get('/ping')
        metrics = client.get('/metrics')

    assert ping.status_code == 200
    assert metrics.status_code == 200
    assert 'http_requests_total' in metrics.text


def test_setup_metrics_disabled():
    app = FastAPI()
    setup_metrics(app, enabled=False, path='/metrics')

    @app.get('/ping')
    async def ping():
        return {'ok': True}

    with TestClient(app) as client:
        resp = client.get('/metrics')

    assert resp.status_code == 404
