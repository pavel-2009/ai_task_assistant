"""Обратносовместимый алиас для единого YOLO-сервиса."""

from .yolo_service import YoloService


class YoloONNXService(YoloService):
    """Совместимая обёртка для инференса YOLO на базе ONNX."""

    def __init__(self):
        super().__init__(provider="onnx")


__all__ = ["YoloONNXService"]
