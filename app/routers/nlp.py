"""
Роутер для обработки запросов, связанных с NLP (Natural Language Processing).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_async_session
from ..error_handlers import AppError
from ..ml.nlp.embedding_service import EmbeddingService
from ..ml.nlp.ner_service import NerService
from ..ml.nlp.semantic_search_service import SemanticSearchService

router = APIRouter(prefix="/nlp", tags=["NLP"])

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


def _get_embedding_service(request: Request) -> EmbeddingService:
    return getattr(request.app.state, "embedding_service", None)


def _get_semantic_search_service(request: Request) -> SemanticSearchService:
    return getattr(request.app.state, "semantic_search_service", None)


def _get_ner_service(request: Request) -> NerService:
    return getattr(request.app.state, "ner_service", None)


@router.post("/embedding", description="Получить эмбеддинг для текста")
async def get_embedding(request: Request, text: str | list[str] = Body(...)):
    """Получить эмбеддинг для текста."""
    normalized_payload = _normalize_texts(text)
    embedding_service = _get_embedding_service(request)

    try:
        if isinstance(normalized_payload, str):
            embedding = await asyncio.to_thread(embedding_service.encode_one, normalized_payload)
        else:
            embedding = await asyncio.to_thread(embedding_service.encode_batch, normalized_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise AppError("Ошибка при построении эмбеддинга", status_code=500) from exc

    return {"embedding": embedding.tolist()}


@router.post(
    "/search",
    description="Искать документы, наиболее похожие на запрос",
    status_code=status.HTTP_200_OK,
)
async def search(
    request: Request,
    query: str = Body(..., embed=True, description="Текст запроса для поиска"),
    top_k: int = Body(5, embed=True, description="Количество результатов для возврата"),
    session: AsyncSession = Depends(get_async_session),
):
    """Поиск документов, наиболее похожих на запрос."""
    normalized_query = _normalize_text(query)
    if top_k <= 0 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k должен быть в диапазоне от 1 до 20")

    semantic_search_service = _get_semantic_search_service(request)

    try:
        results = await semantic_search_service.search(normalized_query, session=session, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise AppError("Ошибка при семантическом поиске", status_code=500) from exc

    return {"results": results}


@router.post(
    "/index",
    description="Индексировать текст, добавляя его эмбеддинг в базу данных",
    status_code=status.HTTP_200_OK,
)
async def index(
    request: Request,
    text: str = Body(..., embed=True, description=""),
    session: AsyncSession = Depends(get_async_session),
):
    """Индексировать текст, добавляя его эмбеддинг в базу данных."""
    normalized_text = _normalize_text(text)
    semantic_search_service = _get_semantic_search_service(request)

    try:
        await semantic_search_service.index(normalized_text, session=session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise AppError("Ошибка индексации текста", status_code=500) from exc

    return {"detail": "Текст успешно индексирован"}


@router.post("/tag-task", description="Получить теги для текста задачи")
async def tag_task(request: Request, text: str = Body(...)):
    """Получить теги для текста задачи."""
    ner_service = _get_ner_service(request)

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

    return {"tags": result}
