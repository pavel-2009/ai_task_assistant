"""
Тесты для NERService
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("spacy")


class FakeEntity:
    def __init__(self, text: str, label: str, confidence: float = 0.8):
        self.text = text
        self.label_ = label
        self._ = SimpleNamespace(confidence=confidence)


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeRuler:
    def add_patterns(self, patterns):
        self.patterns = patterns


class FakeNLP:
    def __init__(self):
        self.pipe_names = []
        self.ruler = FakeRuler()
        self.docs = {
            "Я использую C++ и React.js": FakeDoc([
                FakeEntity("C++", "PRODUCT"),
                FakeEntity("React.js", "PRODUCT"),
            ]),
            "Разрабатываю в Google на Go": FakeDoc([
                FakeEntity("Google", "ORG"),
                FakeEntity("Go", "PRODUCT", 0.91),
            ]),
            "": FakeDoc([]),
            "Использую C++ и Redis": FakeDoc([
                FakeEntity("C++", "PRODUCT"),
                FakeEntity("Redis", "PRODUCT", 0.88),
            ]),
        }

    def add_pipe(self, name, before=None):
        self.pipe_names.append(name)
        return self.ruler

    def get_pipe(self, name):
        return self.ruler

    def __call__(self, text: str):
        return self.docs[text]


@pytest.fixture
def ner_service(monkeypatch):
    from app.ml.nlp.ner_service import NerService

    fake_nlp = FakeNLP()
    monkeypatch.setattr("app.ml.nlp.ner_service.spacy.load", lambda model_name: fake_nlp)
    return NerService()


def test_extract_technologies_special_cases(ner_service):
    technologies = dict(ner_service.extract_technologies("Я использую C++ и React.js"))

    assert technologies["c++"] == 0.95
    assert technologies["react.js"] == 0.95


def test_extract_technologies_with_company(ner_service):
    technologies = dict(ner_service.extract_technologies("Разрабатываю в Google на Go"))

    assert "google" not in technologies
    assert technologies["go"] == 0.91


def test_extract_technologies_empty(ner_service):
    technologies = ner_service.extract_technologies("")
    assert technologies == []


def test_tag_task_format(ner_service):
    result = ner_service.tag_task("Использую C++ и Redis")

    assert "technologies" in result
    assert "confidence" in result
    assert isinstance(result["technologies"], list)
    assert ("c++", 0.95) in result["technologies"]
    assert ("redis", 0.88) in result["technologies"]
    assert 0 <= result["confidence"] <= 1
