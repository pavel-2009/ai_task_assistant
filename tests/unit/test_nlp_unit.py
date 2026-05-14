from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.routers import nlp as nlp_router
from app.ml.nlp.semantic_search_service import SemanticSearchService


@pytest.mark.unit
def test_normalize_text_ok():
    assert nlp_router._normalize_text("  hello  ") == "hello"


@pytest.mark.unit
@pytest.mark.parametrize("bad", [None, 123, "   "])
def test_normalize_text_errors(bad):
    with pytest.raises(HTTPException):
        nlp_router._normalize_text(bad)


@pytest.mark.unit
def test_normalize_texts_list_limits():
    with pytest.raises(HTTPException):
        nlp_router._normalize_texts([])

    too_many = ["x"] * 11
    with pytest.raises(HTTPException):
        nlp_router._normalize_texts(too_many)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tag_task_success_extracts_only_names():
    class _Ner:
        def tag_task(self, text):
            return {"technologies": [("python", 0.99), ("fastapi", 0.95)]}

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ner_service=_Ner())))

    response = await nlp_router.tag_task(request=request, text="Build API with Python and FastAPI")

    assert response.tags == ["python", "fastapi"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tag_task_service_not_initialized():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ner_service=None)))

    with pytest.raises(HTTPException) as exc:
        await nlp_router.tag_task(request=request, text="Any text")

    assert exc.value.status_code == 503


@pytest.mark.unit
@pytest.mark.asyncio
async def test_semantic_search_service_delete_calls_clear_cache():
    vector_db = SimpleNamespace(delete=AsyncMock())

    service = SemanticSearchService(
        embedding_service=SimpleNamespace(dimension=4),
        vector_db=vector_db,
        redis_client=None,
    )
    service.clear_cache = AsyncMock()

    await service.delete(123)

    vector_db.delete.assert_awaited_once_with("123")
    service.clear_cache.assert_awaited_once()
