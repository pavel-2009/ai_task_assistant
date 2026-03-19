"""Object detection services and YOLO training assets."""

from .yolo_onnx_service import YoloONNXService
from .yolo_service import YoloService

__all__ = ["YoloONNXService", "YoloService"]
