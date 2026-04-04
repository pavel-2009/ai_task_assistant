"""Юнит-тесты реальных NLP сервисов (без mock): NER и Embedding."""

from __future__ import annotations

import numpy as np

from app.ml.nlp.embedding_service import EmbeddingService
from app.ml.nlp.ner_service import NerService


def test_real_embedding_service_encode_one_returns_normalized_vector():
    """Реальный EmbeddingService должен вернуть вектор нужной размерности."""
    service = EmbeddingService()
    emb = service.encode_one("Build FastAPI service with PostgreSQL")

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (service.dimension,)
    assert np.isfinite(emb).all()
    assert np.linalg.norm(emb) > 0


def test_real_ner_service_extracts_technology_entities():
    """Реальный NerService должен извлекать технологии из текста."""
    service = NerService()
    result = service.tag_task("Need to develop backend using FastAPI and PostgreSQL")

    technologies = [name for name, _conf in result["technologies"]]
    assert "fastapi" in technologies
    assert "postgresql" in technologies
    assert 0 <= result["confidence"] <= 1
