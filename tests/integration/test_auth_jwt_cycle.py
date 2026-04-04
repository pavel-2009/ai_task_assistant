from __future__ import annotations

import os
import time
import uuid

import jwt
import pytest


def _creds(prefix: str = "it_user") -> dict[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return {"username": f"{prefix}_{suffix}", "password": "Password1!"}


@pytest.mark.integration
def test_registration_and_auth_edge_cases(integration_httpx_client_a):
    creds = _creds()

    register = integration_httpx_client_a.post("/auth/register", json=creds)
    assert register.status_code == 201, register.text

    duplicate = integration_httpx_client_a.post("/auth/register", json=creds)
    assert duplicate.status_code == 400

    weak_password = integration_httpx_client_a.post(
        "/auth/register",
        json={"username": f"weak_{uuid.uuid4().hex[:6]}", "password": "weak"},
    )
    assert weak_password.status_code in (400, 422)

    missing_username = integration_httpx_client_a.post(
        "/auth/register",
        json={"username": "", "password": "Password1!"},
    )
    assert missing_username.status_code in (400, 422)

    wrong_password = integration_httpx_client_a.post(
        "/auth/login",
        data={"username": creds["username"], "password": "WrongPass1!"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert wrong_password.status_code == 400


@pytest.mark.integration
def test_jwt_full_lifecycle(integration_httpx_client_a):
    creds = _creds("jwt_user")

    reg = integration_httpx_client_a.post("/auth/register", json=creds)
    assert reg.status_code == 201, reg.text

    login = integration_httpx_client_a.post(
        "/auth/login",
        data=creds,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]

    create_task = integration_httpx_client_a.post(
        "/tasks/",
        json={"title": "jwt-task", "description": "JWT protected endpoint check"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert create_task.status_code == 201, create_task.text

    without_token = integration_httpx_client_a.post(
        "/tasks/",
        json={"title": "jwt-task-2", "description": "no token"},
    )
    assert without_token.status_code == 401

    malformed = integration_httpx_client_a.post(
        "/tasks/",
        json={"title": "jwt-task-3", "description": "bad token"},
        headers={"Authorization": "Bearer not.a.valid.token"},
    )
    assert malformed.status_code == 401

    expired_token = jwt.encode(
        {"user_id": create_task.json()["author_id"], "exp": int(time.time()) - 60},
        os.getenv("SECRET_KEY", "test-secret-key"),
        algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
    )
    expired = integration_httpx_client_a.post(
        "/tasks/",
        json={"title": "jwt-task-4", "description": "expired token"},
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert expired.status_code == 401
