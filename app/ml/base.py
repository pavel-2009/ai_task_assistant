"""Общие базовые интерфейсы для ML-сервисов инференса."""

from __future__ import annotations


class BaseMLService:
    """Минимальный интерфейс для сервисов инференса."""

    def predict(self, data):
        raise NotImplementedError
