"""API-тесты загрузки аватаров и CV/Celery сценариев."""

import io
import os

import cv2
import numpy as np
import pytest

from app.celery_app import celery_app
from app.core.config import config


@pytest.fixture
def sample_jpeg_bytes():
    """Генерирует валидное JPEG-изображение в памяти."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    ok, img_encoded = cv2.imencode(".jpg", img)
    assert ok
    return io.BytesIO(img_encoded.tobytes())


@pytest.fixture
def celery_eager_mode():
    """Включает eager-режим Celery для тестов submit/status без отдельного воркера."""
    original_always_eager = celery_app.conf.task_always_eager
    original_store_eager = celery_app.conf.task_store_eager_result
    original_backend = celery_app.conf.result_backend

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_store_eager_result = True
    celery_app.conf.result_backend = config.CELERY_BROKER_URL

    yield

    celery_app.conf.task_always_eager = original_always_eager
    celery_app.conf.task_store_eager_result = original_store_eager
    celery_app.conf.result_backend = original_backend


@pytest.mark.asyncio
async def test_upload_avatar_unauthorized(client, sample_jpeg_bytes):
    """Тестирование загрузки аватара без авторизации."""
    response = client.post(
        "/tasks/1/avatar",
        files={"image": ("avatar.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_upload_avatar(authorized_client, create_base_tasks, sample_jpeg_bytes):
    """Тестирование успешной загрузки аватара."""
    response = authorized_client.post(
        "/tasks/1/avatar",
        files={"image": ("avatar.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json().get("filename").startswith("1_")

    filepath = response.json().get("filepath")
    assert filepath is not None
    assert filepath.endswith(".jpeg")
    assert os.path.exists(filepath)


@pytest.mark.asyncio
async def test_cv_predict_submit_and_status_success(
    authorized_client,
    create_base_tasks,
    sample_jpeg_bytes,
    celery_eager_mode,
):
    """Проверяет submit/status для предсказания класса через API + Celery."""
    upload_response = authorized_client.post(
        "/tasks/1/avatar",
        files={"image": ("avatar.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert upload_response.status_code == 200

    submit_response = authorized_client.post("/tasks/1/predict/submit")
    assert submit_response.status_code == 202
    celery_task_id = submit_response.json()["celery_task_id"]

    status_response = authorized_client.get(f"/tasks/1/predict/status/{celery_task_id}")
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] in {"SUCCESS", "FAILURE"}


@pytest.mark.asyncio
async def test_cv_detect_submit_and_status_success(
    authorized_client,
    create_base_tasks,
    sample_jpeg_bytes,
    celery_eager_mode,
):
    """Проверяет submit/status для детекции объектов через API + Celery."""
    upload_response = authorized_client.post(
        "/tasks/1/avatar",
        files={"image": ("avatar.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert upload_response.status_code == 200

    submit_response = authorized_client.post("/tasks/1/detect/submit")
    assert submit_response.status_code == 202
    celery_task_id = submit_response.json()["celery_task_id"]

    status_response = authorized_client.get(f"/tasks/1/detect/status/{celery_task_id}")
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] in {"SUCCESS", "FAILURE"}


@pytest.mark.asyncio
async def test_cv_segment_submit_and_status_success(
    authorized_client,
    create_base_tasks,
    sample_jpeg_bytes,
    celery_eager_mode,
):
    """Проверяет submit/status для сегментации через API + Celery."""
    upload_response = authorized_client.post(
        "/tasks/1/avatar",
        files={"image": ("avatar.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert upload_response.status_code == 200

    submit_response = authorized_client.post("/tasks/1/segment/submit")
    assert submit_response.status_code == 202
    celery_task_id = submit_response.json()["celery_task_id"]

    status_response = authorized_client.get(f"/tasks/1/segment/status/{celery_task_id}")
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["status"] in {"SUCCESS", "FAILURE"}
