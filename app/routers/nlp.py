"""
Роутер для обработки запросов, связанных с NLP (Natural Language Processing).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..core import config
from ..db import get_async_session
from ..error_handlers import AppError
from ..ml.nlp.embedding_service import EmbeddingService
from ..ml.nlp.ner_service import NerService
from ..ml.nlp.semantic_search_service import SemanticSearchService
from ..schemas import (
    EmbeddingResponse, SearchResults, IndexResponse, NLPTagTaskResponse
)
from ..services import (
    get_embedding as get_embedding_service_dependency,
    get_semantic_search as get_semantic_search_service_dependency,
    get_ner as get_ner_service_dependency,
)

router = APIRouter(prefix="/nlp", tags=["NLP"])
logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 10
MAX_TEXT_LENGTH = 1000


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Текст должен быть строкой")

    normalized_text = text.strip()
    if not normalized_text:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    if len(normalized_text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Текст слишком длинный. Максимум {MAX_TEXT_LENGTH} символов",
        )

    return normalized_text


def _normalize_texts(payload: str | list[str]) -> str | list[str]:
    if isinstance(payload, str):
        return _normalize_text(payload)
    if isinstance(payload, list):
        if not payload:
            raise HTTPException(status_code=400, detail="Список текстов не может быть пустым")
        if len(payload) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Слишком много текстов в списке. Максимум {MAX_BATCH_SIZE}",
            )
        return [_normalize_text(item) for item in payload]

    raise HTTPException(
        status_code=400,
        detail="Неверный формат данных. Ожидается строка или список строк",
    )


@router.post("/embedding", description="Получить эмбеддинг для текста", response_model=EmbeddingResponse)
async def get_embedding(
    text: str | list[str] = Body(...),
    embedding_service: EmbeddingService = Depends(get_embedding_service_dependency),
):
    """Получить эмбеддинг для текста."""
    normalized_payload = _normalize_texts(text)

    try:
        if isinstance(normalized_payload, str):
            embedding = await asyncio.to_thread(embedding_service.encode_one, normalized_payload)
        else:
            embedding = await asyncio.to_thread(embedding_service.encode_batch, normalized_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise AppError("Ошибка при построении эмбеддинга", status_code=500) from exc

    return EmbeddingResponse(embedding=embedding.tolist())


@router.post(
    "/search",
    description="Искать документы, наиболее похожие на запрос",
    status_code=status.HTTP_200_OK,
    response_model=SearchResults
)
async def search(
    query: str = Body(..., embed=True, description="Текст запроса для поиска"),
    top_k: int = Body(config.DEFAULT_TOP_K, embed=True, description="Количество результатов для возврата"),
    semantic_search_service: SemanticSearchService = Depends(get_semantic_search_service_dependency),
    session: AsyncSession = Depends(get_async_session),
):
    """Поиск документов, наиболее похожих на запрос."""
    normalized_query = _normalize_text(query)
    if top_k <= 0 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k должен быть в диапазоне от 1 до 20")

    try:
        results = await semantic_search_service.search(normalized_query, session=session, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Ошибка при семантическом поиске: {exc}", exc_info=True)
        raise AppError("Ошибка при семантическом поиске", status_code=500) from exc

    return SearchResults(results=results, total=len(results))


@router.post(
    "/index",
    description="Индексировать текст, добавляя его эмбеддинг в базу данных",
    status_code=status.HTTP_200_OK,
    response_model=IndexResponse
)
async def index(
    text: str = Body(..., embed=True, description=""),
    semantic_search_service: SemanticSearchService = Depends(get_semantic_search_service_dependency),
    session: AsyncSession = Depends(get_async_session),
):
    """Индексировать текст, добавляя его эмбеддинг в базу данных."""
    normalized_text = _normalize_text(text)
    try:
        await semantic_search_service.index(normalized_text, session=session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise AppError("Ошибка индексации текста", status_code=500) from exc

    return IndexResponse(detail="Текст успешно индексирован")


@router.post("/tag-task", description="Получить теги для текста задачи", response_model=NLPTagTaskResponse)
async def tag_task(
    text: str = Body(...),
    ner_service: NerService = Depends(get_ner_service_dependency),
):
    """Получить теги для текста задачи."""
    if ner_service is None:
        raise HTTPException(
            status_code=503,
            detail="NerService не инициализирован. Проверьте логи приложения",
        )

    normalized_text = _normalize_text(text)

    try:
        result = await asyncio.to_thread(ner_service.tag_task, normalized_text)
    except Exception as exc:
        raise AppError("Ошибка при обработке текста NER сервисом", status_code=500) from exc

    technologies = result.get("technologies", []) if isinstance(result, dict) else []
    tags = [name for name, _confidence in technologies]
    return NLPTagTaskResponse(tags=tags)
