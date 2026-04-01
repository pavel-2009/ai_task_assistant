"""Централизованные обработчики ошибок FastAPI."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Контролируемая ошибка приложения."""

    def __init__(self, detail: str, status_code: int = 500, code: str = "INTERNAL_ERROR"):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.code = code


def register_exception_handlers(app: FastAPI) -> None:
    """Регистрирует общие обработчики ошибок."""

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "detail": exc.detail}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"code": "VALIDATION_ERROR", "detail": "Invalid request parameters"}
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"code": "INTERNAL_ERROR", "detail": "Внутренняя ошибка сервера"}
        )
