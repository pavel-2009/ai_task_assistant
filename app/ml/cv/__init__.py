"""Computer vision package split by task domain."""

from .classification import InferenceService
from .detection import YoloONNXService, YoloService
from .segmentation import SegmentationService

__all__ = [
    "InferenceService",
    "SegmentationService",
    "YoloONNXService",
    "YoloService",
]
