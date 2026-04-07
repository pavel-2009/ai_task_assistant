"""Backward-compatible alias for the unified YOLO service."""

from .yolo_service import YoloService


class YoloONNXService(YoloService):
    """Compatibility wrapper for ONNX-based YOLO inference."""

    def __init__(self):
        super().__init__(provider="onnx")


__all__ = ["YoloONNXService"]
