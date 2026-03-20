"""
ONNX-сервис для YOLOv8 с возвратом имени класса (обратная совместимость)

Это модуль для обеспечения обратной совместимости.
Основная реализация находится в пакете yolo_onnx/
"""

from .yolo_onnx import YoloONNXService

__all__ = ['YoloONNXService']
