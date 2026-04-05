from __future__ import annotations

import io

import pytest
from PIL import Image


def _make_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), color=(30, 144, 255))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.integration
def test_avatar_upload_and_all_analysis_types(integration_httpx_client_a, integration_auth_headers_a):
    task_resp = integration_httpx_client_a.post(
        "/tasks/",
        json={"title": "avatar-flow", "description": "avatar test pipeline"},
        headers=integration_auth_headers_a,
    )
    assert task_resp.status_code == 201, task_resp.text
    task_id = task_resp.json()["id"]

    upload_resp = integration_httpx_client_a.post(
        f"/tasks/{task_id}/avatar",
        files={"image": ("avatar.jpg", _make_jpeg_bytes(), "image/jpeg")},
        headers=integration_auth_headers_a,
    )
    assert upload_resp.status_code == 200, upload_resp.text

    predict_submit = integration_httpx_client_a.post(
        f"/tasks/{task_id}/predict/submit",
        headers=integration_auth_headers_a,
    )
    detect_submit = integration_httpx_client_a.post(
        f"/tasks/{task_id}/detect/submit",
        headers=integration_auth_headers_a,
    )
    segment_submit = integration_httpx_client_a.post(
        f"/tasks/{task_id}/segment/submit",
        headers=integration_auth_headers_a,
    )

    assert predict_submit.status_code == 202, predict_submit.text
    assert detect_submit.status_code == 202, detect_submit.text
    assert segment_submit.status_code == 202, segment_submit.text

    predict_status = integration_httpx_client_a.get(
        f"/tasks/{task_id}/predict/status/{predict_submit.json()['celery_task_id']}",
        headers=integration_auth_headers_a,
    )
    detect_status = integration_httpx_client_a.get(
        f"/tasks/{task_id}/detect/status/{detect_submit.json()['celery_task_id']}",
        headers=integration_auth_headers_a,
    )
    segment_status = integration_httpx_client_a.get(
        f"/tasks/{task_id}/segment/status/{segment_submit.json()['celery_task_id']}",
        headers=integration_auth_headers_a,
    )

    assert predict_status.status_code == 200, predict_status.text
    assert detect_status.status_code == 200, detect_status.text
    assert segment_status.status_code == 200, segment_status.text

    assert predict_status.json()["status"] in {"PENDING", "STARTED", "SUCCESS", "FAILURE"}
    assert detect_status.json()["status"] in {"PENDING", "STARTED", "SUCCESS", "FAILURE"}
    assert segment_status.json()["status"] in {"PENDING", "STARTED", "SUCCESS", "FAILURE"}


@pytest.mark.integration
def test_avatar_upload_edge_cases(integration_httpx_client_a, integration_auth_headers_a):
    task_resp = integration_httpx_client_a.post(
        "/tasks/",
        json={"title": "avatar-edge", "description": "avatar edge cases"},
        headers=integration_auth_headers_a,
    )
    assert task_resp.status_code == 201, task_resp.text
    task_id = task_resp.json()["id"]

    no_file = integration_httpx_client_a.post(
        f"/tasks/{task_id}/avatar",
        headers=integration_auth_headers_a,
    )
    assert no_file.status_code == 400

    invalid_file = integration_httpx_client_a.post(
        f"/tasks/{task_id}/avatar",
        files={"image": ("avatar.txt", b"not-an-image", "text/plain")},
        headers=integration_auth_headers_a,
    )
    assert invalid_file.status_code == 400
