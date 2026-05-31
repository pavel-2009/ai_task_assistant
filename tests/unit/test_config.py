import pytest
from pydantic import ValidationError

from app.core.config import Settings


BASE = {
    'SECRET_KEY': 'secret',
    'DATABASE_URL': 'sqlite+aiosqlite:///./test.db',
    'REDIS_URL': 'redis://localhost:6379/0',
    'CELERY_BROKER_URL': 'redis://localhost:6379/1',
    'CELERY_RESULT_BACKEND': 'redis://localhost:6379/2',
    'LLM_API_KEY': 'test-key',
}


def test_default_values():
    settings = Settings(**BASE)
    assert settings.JWT_EXPIRE_MINUTES == 1
    assert settings.JWT_ALGORITHM == 'HS256'


def test_jwt_expire_minutes_zero_validation():
    with pytest.raises(ValidationError):
        Settings(**BASE, JWT_EXPIRE_MINUTES=0)


def test_jwt_expire_minutes_over_1440_validation():
    with pytest.raises(ValidationError):
        Settings(**BASE, JWT_EXPIRE_MINUTES=1441)


def test_empty_secret_key_validation():
    with pytest.raises(ValidationError):
        BASE["SECRET_KEY"] = ""
        Settings(**BASE)


def test_whitespace_secret_key_validation():
    with pytest.raises(ValidationError):
        BASE["SECRET_KEY"] = " "
        Settings(**BASE)
