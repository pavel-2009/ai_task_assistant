"""
Конфигурация и константы для YOLO ONNX сервиса
"""

from pathlib import Path
import os
import onnxruntime as ort

# Пути
ONNX_WEIGHTS_PATH = Path(__file__).parent.parent / "yolov8n.onnx"
VISUALIZATION_DIR = Path(__file__).parent.parent.parent.parent / 'avatars' / 'visualizations'

# COCO class names
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# ONNX Runtime параметры
def get_session_options() -> ort.SessionOptions:
    """Создание оптимизированных опций сессии ONNX Runtime"""
    sess_options = ort.SessionOptions() # Включаем оптимизации для ускорения инференса
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # Включаем все оптимизации
    sess_options.intra_op_num_threads = os.cpu_count() # Используем все доступные ядра для оптимизации
    sess_options.inter_op_num_threads = 1 # Устанавливаем 1 для последовательного выполнения, что может быть быстрее для небольших моделей
    sess_options.enable_cpu_mem_arena = False  # Отключаем для скорости
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL # Последовательное выполнение может быть быстрее для небольших моделей
    return sess_options

# Параметры инференса
DEFAULT_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", 0.35))
DEFAULT_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", 0.55))

# Параметры кэширования
CACHE_SIZE = 5
METRICS_WINDOW = 100

# Параметры обработки
THREADPOOL_WORKERS = 2
