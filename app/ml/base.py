"""Shared base interfaces for ML inference services."""

from __future__ import annotations


class BaseMLService:
    """Minimal interface for inference services."""

    def predict(self, data):
        raise NotImplementedError
