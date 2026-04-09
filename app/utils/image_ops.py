"""
Операции с изображениями
"""

import cv2
import numpy as np

from app.core import config


def validate_image(image_bytes: bytes) -> bool:
    """Проверка валидности изображения"""

    if not image_bytes:
        return False

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    if image_array.size == 0:
        return False

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return False

    return True


def resize_image(image_bytes: bytes, max_size: int = config.MAX_IMAGE_SIZE_PX) -> bytes:
    """Изменение размера изображения, если оно превышает max_size по ширине или высоте"""

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]

    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    image_resized = cv2.imencode(".jpeg", img)[1].tobytes()

    return image_resized
