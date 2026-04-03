"""Простые API-тесты загрузки аватаров."""

import io

import os

import cv2
import numpy as np
import pytest


@pytest.mark.asyncio
async def test_upload_avatar_unauthorized(client):
    """Тестирование загрузки аватара без авторизации."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    
    response = client.post("/tasks/1/avatar", files={"image": ("avatar.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 401
    

@pytest.mark.asyncio
async def test_upload_avatar(authorized_client, create_base_tasks):

    # Создаем тестовое изображение
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    
    response = authorized_client.post("/tasks/1/avatar", files={"image": ("avatar.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200
    assert response.json().get("filename").startswith(f"1_")
    
    filepath = response.json().get("filepath")
    
    assert filepath is not None
    assert filepath.startswith("avatars/")
    assert filepath.endswith(".jpeg")
    
    assert os.path.exists(filepath)
