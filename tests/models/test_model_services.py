"""
Тесты прямого вызова ML-сервисов (без API-роутеров).

Запускать при установленных ML-зависимостях.
"""

from pathlib import Path
import io
import sys

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _build_image_bytes() -> bytes:
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (320, 320), color=(80, 120, 160))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.models
def test_inference_service_predict_raw():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from app.ml.cv.classification.inference_service import InferenceService

    checkpoint_path = Path("checkpoints/model.pth")
    if not checkpoint_path.exists():
        pytest.skip("checkpoint checkpoints/model.pth not found")

    service = InferenceService(
        checkpoints_path=checkpoint_path,
        idx_to_class={0: "cat", 1: "dog", 2: "house"},
    )

    result = service.predict(_build_image_bytes())

    assert isinstance(result, dict)
    assert "class_id" in result
    assert "class_name" in result
    assert "confidence" in result
    assert isinstance(result["class_id"], int)
    assert isinstance(result["class_name"], str)
    assert 0.0 <= float(result["confidence"]) <= 1.0


@pytest.mark.models
def test_yolo_service_predict_raw():
    pytest.importorskip("ultralytics")
    pytest.importorskip("cv2")

    from app.ml.cv.detection.yolo_service import YoloService

    model_path = Path("app/ml/cv/detection/yolov8n.pt")
    if not model_path.exists():
        pytest.skip("YOLO weights app/ml/cv/detection/yolov8n.pt not found")

    service = YoloService(str(model_path))
    result = service.predict(_build_image_bytes())

    assert isinstance(result, list)
    if result:
        assert isinstance(result[0], dict)
        assert "class_name" in result[0]
        assert "confidence" in result[0]
        assert "box" in result[0]


@pytest.mark.models
def test_yolo_onnx_service_predict_raw():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("cv2")

    from app.ml.cv.detection.yolo_onnx_service import YoloONNXService

    model_path = Path("app/ml/cv/detection/yolov8n.onnx")
    if not model_path.exists():
        pytest.skip("ONNX weights app/ml/cv/detection/yolov8n.onnx not found")

    service = YoloONNXService(str(model_path))
    result = service.predict(_build_image_bytes())

    assert isinstance(result, list)
    if result:
        assert isinstance(result[0], dict)
        assert "class_name" in result[0]
        assert "confidence" in result[0]
        assert "box" in result[0]


@pytest.mark.models
def test_segmentation_service_predict_raw():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")

    from app.ml.cv.segmentation.segmentation_service import SegmentationService

    service = SegmentationService()
    result = service.segment_image(_build_image_bytes())

    assert isinstance(result, (bytes, bytearray))
    assert len(result) > 0
