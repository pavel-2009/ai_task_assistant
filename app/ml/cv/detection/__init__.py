"""Сервисы детекции объектов и ресурсы обучения YOLO."""

from .yolo_onnx_service import YoloONNXService
from .yolo_service import YoloService

__all__ = ["YoloONNXService", "YoloService"]
