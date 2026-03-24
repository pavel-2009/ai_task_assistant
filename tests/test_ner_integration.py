"""
Интеграционные тесты для NerService
"""

import pytest
from typing import Any


pytestmark = pytest.mark.models


@pytest.fixture
def ner_service() -> Any:
    spacy = pytest.importorskip("spacy")

    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model 'en_core_web_sm' is not installed")

    from app.ml.nlp.ner_service import NerService

    return NerService(model_name="en_core_web_sm")


def _tech_map(technologies: list[tuple[str, float]]) -> dict[str, float]:
    return {name.lower(): confidence for name, confidence in technologies}


def test_ner_extracts_technologies(ner_service: Any):
    text = "Мы разрабатываем на Python с использованием FastAPI и PostgreSQL"

    technologies = _tech_map(ner_service.extract_technologies(text))

    assert "python" not in technologies
    assert "fastapi" in technologies
    assert "postgresql" in technologies

    for confidence in technologies.values():
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


def test_ner_handles_empty_text(ner_service: Any):
    result = ner_service.tag_task("")

    assert result == {"technologies": [], "confidence": 0}


def test_ner_blacklist_filtering(ner_service: Any):
    text = "Мы используем Apple для разработки"

    technologies = _tech_map(ner_service.extract_technologies(text))

    assert "apple" not in technologies
    # If support for phrases like "Apple Swift" is added later, this case may
    # need to be revisited because the company and the technology can overlap.


def test_ner_special_cases(ner_service: Any):
    text = "Нужен C++ разработчик и React.js специалист"

    technologies = _tech_map(ner_service.extract_technologies(text))

    assert "c++" in technologies
    assert "react.js" in technologies
    assert technologies["c++"] == 0.95
    assert technologies["react.js"] == 0.95
