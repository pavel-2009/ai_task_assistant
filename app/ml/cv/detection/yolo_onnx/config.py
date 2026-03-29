"""
Конфигурация и константы для YOLO ONNX сервиса
"""

from pathlib import Path
import os

import onnxruntime as ort

from app.core import config

# Пути
ONNX_WEIGHTS_PATH = Path(__file__).parent.parent / "yolov8n.onnx"
VISUALIZATION_DIR = Path(__file__).parent.parent.parent.parent / "avatars" / "visualizations"

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_session_options() -> ort.SessionOptions:
    """Создание оптимизированных опций сессии ONNX Runtime"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count()
    sess_options.inter_op_num_threads = 1
    sess_options.enable_cpu_mem_arena = False
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return sess_options


DEFAULT_CONF_THRESHOLD = config.YOLO_CONF_THRESHOLD
DEFAULT_IOU_THRESHOLD = config.YOLO_IOU_THRESHOLD
CACHE_SIZE = config.FRAME_CACHE_SIZE
METRICS_WINDOW = 100
THREADPOOL_WORKERS = 2
