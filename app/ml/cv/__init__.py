"""Пакет компьютерного зрения, разделённый по задачам."""

from .classification import InferenceService
from .detection import YoloONNXService, YoloService
from .segmentation import SegmentationService

__all__ = [
    "InferenceService",
    "SegmentationService",
    "YoloONNXService",
    "YoloService",
]
