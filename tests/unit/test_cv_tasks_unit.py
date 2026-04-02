from pathlib import Path

import pytest

from app.ml.cv import tasks as cv_tasks


@pytest.mark.unit
def test_predict_avatar_class_success(tmp_path, monkeypatch):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"img-bytes")

    class _Inference:
        def predict(self, image_bytes):
            assert image_bytes == b"img-bytes"
            return {"predicted_class": "cat"}

    monkeypatch.setattr(cv_tasks, "get_inference", lambda: _Inference())

    result = cv_tasks.predict_avatar_class(task_id=1, image_path=str(image_path))

    assert result == {"predicted_class": "cat"}


@pytest.mark.unit
def test_predict_avatar_class_runtime_error(monkeypatch):
    def _broken():
        raise RuntimeError("service not ready")

    monkeypatch.setattr(cv_tasks, "get_inference", _broken)

    result = cv_tasks.predict_avatar_class(task_id=1, image_path="/tmp/missing.jpg")

    assert "error" in result
    assert "service not ready" in result["error"]


@pytest.mark.unit
def test_detect_and_visualize_task_success(tmp_path, monkeypatch):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"img-bytes")

    class _Yolo:
        def predict_and_visualize(self, image_bytes, task_id):
            assert image_bytes == b"img-bytes"
            assert task_id == 55
            return [{"label": "cat", "score": 0.9}], Path("avatars/55_detected.jpg")

    monkeypatch.setattr(cv_tasks, "get_yolo", lambda: _Yolo())

    result = cv_tasks.detect_and_visualize_task(task_id=55, image_path=str(image_path))

    assert result["detections"][0]["label"] == "cat"
    assert result["visualized_image"] == "avatars/55_detected.jpg"


@pytest.mark.unit
def test_segment_image_task_saves_output(tmp_path, monkeypatch):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"source")

    class _Segmentation:
        def segment_image(self, image_bytes):
            assert image_bytes == b"source"
            return b"segmented-bytes"

    monkeypatch.setattr(cv_tasks, "get_segmentation", lambda: _Segmentation())
    monkeypatch.setattr(cv_tasks, "Path", lambda _p: tmp_path / "segments")

    result = cv_tasks.segment_image_task(task_id=3, image_path=str(image_path))

    saved_file = tmp_path / "segments" / "3_segmentation.png"
    assert result == b"segmented-bytes"
    assert saved_file.exists()
    assert saved_file.read_bytes() == b"segmented-bytes"
