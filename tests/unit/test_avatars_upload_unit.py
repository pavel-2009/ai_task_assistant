"""Простые unit-тесты утилит изображений для аватаров."""

import cv2
import numpy as np

from app.utils.image_ops import resize_image, validate_image


def _make_image_bytes(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_validate_image_ok():
    assert validate_image(_make_image_bytes(64, 64)) is True


def test_validate_image_bad_bytes():
    assert validate_image(b"not-an-image") is False


def test_resize_image_returns_jpeg_bytes():
    result = resize_image(_make_image_bytes(3000, 2000), max_size=512)
    assert isinstance(result, bytes)
    assert validate_image(result) is True
