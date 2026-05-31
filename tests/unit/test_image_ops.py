import cv2
import numpy as np
import pytest

from app.utils.image_ops import resize_image, validate_image


def _make_image_bytes(fmt: str, size: tuple[int, int] = (64, 64)) -> bytes:
    img = np.full((size[1], size[0], 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(fmt, img)
    assert ok
    return encoded.tobytes()


def test_validate_jpeg_image():
    assert validate_image(_make_image_bytes('.jpg')) is True


def test_validate_png_image():
    assert validate_image(_make_image_bytes('.png')) is True


def test_validate_invalid_data():
    assert validate_image(b'not_an_image') is False


def test_resize_small_image():
    image_bytes = _make_image_bytes('.jpg', size=(100, 100))
    resized = resize_image(image_bytes, max_size=200)
    img = cv2.imdecode(np.frombuffer(resized, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (100, 100)


def test_resize_large_image():
    image_bytes = _make_image_bytes('.jpg', size=(2000, 1200))
    resized = resize_image(image_bytes, max_size=1024)
    img = cv2.imdecode(np.frombuffer(resized, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 1024


def test_empty_file_boundary_case():
    assert validate_image(b'') is False
    with pytest.raises(Exception):
        resize_image(b'')
