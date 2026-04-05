from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.error_handlers import AppError, register_exception_handlers


def test_app_error_different_codes():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get('/teapot')
    async def teapot():
        raise AppError('teapot', status_code=418, code='TEAPOT')

    @app.get('/bad')
    async def bad():
        raise AppError('bad request', status_code=400, code='BAD_REQUEST')

    with TestClient(app) as client:
        resp_teapot = client.get('/teapot')
        resp_bad = client.get('/bad')

    assert resp_teapot.status_code == 418
    assert resp_teapot.json() == {'code': 'TEAPOT', 'detail': 'teapot'}
    assert resp_bad.status_code == 400
    assert resp_bad.json() == {'code': 'BAD_REQUEST', 'detail': 'bad request'}


def test_http_exception_passthrough():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get('/http')
    async def http_error():
        raise HTTPException(status_code=404, detail='not found')

    with TestClient(app) as client:
        resp = client.get('/http')

    assert resp.status_code == 404
    assert resp.json() == {'detail': 'not found'}


def test_validation_error_to_422():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get('/items/{item_id}')
    async def get_item(item_id: int):
        return {'item_id': item_id}

    with TestClient(app) as client:
        resp = client.get('/items/not-int')

    assert resp.status_code == 422
    assert resp.json()['code'] == 'VALIDATION_ERROR'


def test_unhandled_exception_to_500():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get('/boom')
    async def boom():
        raise RuntimeError('boom')

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get('/boom')

    assert resp.status_code == 500
    assert resp.json() == {'code': 'INTERNAL_ERROR', 'detail': 'Внутренняя ошибка сервера'}


def test_error_format_contains_code_and_detail():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get('/err')
    async def err():
        raise AppError('message', status_code=409, code='CONFLICT')

    with TestClient(app) as client:
        resp = client.get('/err')

    payload = resp.json()
    assert set(payload.keys()) == {'code', 'detail'}
