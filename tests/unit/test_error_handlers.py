from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.error_handlers import AppError, register_exception_handlers


def test_app_error_handler_returns_structured_json():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom():
        raise AppError("custom error", status_code=418, code="TEAPOT")

    with TestClient(app) as client:
        resp = client.get("/boom")

    assert resp.status_code == 418
    assert resp.json() == {"code": "TEAPOT", "detail": "custom error"}


def test_validation_error_handler_returns_422():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/items/{item_id}")
    async def get_item(item_id: int):
        return {"item_id": item_id}

    with TestClient(app) as client:
        resp = client.get("/items/not-int")

    assert resp.status_code == 422
    assert resp.json()["code"] == "VALIDATION_ERROR"
