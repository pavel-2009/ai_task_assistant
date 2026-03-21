"""
Тесты для NERService
"""

import pytest

from app.ml.nlp.ner_service import NerService


@pytest.fixture
def ner_service():
    return NerService()


def test_extract_technologies_simple(ner_service):
    """Тест извлечения простых технологий"""
    text = "Я использую FastAPI и PostgreSQL"
    technologies = ner_service.extract_technologies(text)
    
    assert "fastapi" in technologies
    assert "postgresql" in technologies


def test_extract_technologies_with_company(ner_service):
    """Тест: компания не должна попадать в технологии"""
    text = "Разрабатываю в Google на Go"
    technologies = ner_service.extract_technologies(text)
    
    assert "google" not in technologies
    assert "go" in technologies  # Go - язык программирования


def test_extract_technologies_empty(ner_service):
    """Тест на пустой текст"""
    technologies = ner_service.extract_technologies("")
    assert technologies == []


def test_tag_task_format(ner_service):
    """Тест формата ответа tag_task"""
    result = ner_service.tag_task("Использую FastAPI и Redis")
    
    assert "technologies" in result
    assert "confidence" in result
    assert isinstance(result["technologies"], list)
    assert 0 <= result["confidence"] <= 1