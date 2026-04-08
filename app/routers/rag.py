"""
Роутер для запросов к RAG модели.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_async_session
from ..error_handlers import AppError
from ..schemas import AskRequest, AskResponse
from app.ml.nlp.tasks import reindex_tasks as reindex_tasks_task

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=['RAG']
)


@router.post("/reindex", description="Переиндексация задач в RAG API")
async def reindex_tasks_endpoint(
    request: Request,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Ручной триггер для переиндексации задач в RAG API.
    """
    try:
        reindex_tasks_task.delay()  # Запускаем переиндексацию в фоновом режиме
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Переиндексация задач запущена"})
    except Exception as exc:
        logger.error(f"Ошибка при переиндексации RAG: {exc}", exc_info=True)
        raise AppError("Ошибка при переиндексации RAG", status_code=500) from exc


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: Request,
    body: AskRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Запрос к RAG модели.
    """
    rag_service = getattr(request.app.state, "rag_service", None)

    if not rag_service:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при загрузке RAG-модели"
        )

    try:
        result = await rag_service.ask(
            body.query,
            session,
            body.top_k,
            body.use_cache
        )
        # Убеждаемся, что ответ соответствует схеме AskResponse
        if isinstance(result, dict):
            return AskResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources"),
                confidence=result.get("confidence"),
                cached=result.get("cached"),
            )
        return result
    except Exception as exc:
        logger.error(f"Ошибка при обработке RAG запроса: {exc}", exc_info=True)
        raise AppError("Ошибка при обработке RAG запроса", status_code=500) from exc


@router.post("/ask/stream", description="Потоковый SSE-запрос к RAG модели, returns text/event-stream")
async def ask_stream(
    request: Request,
    body: AskRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Потоковый SSE-запрос к RAG модели.
    Возвращает поток данных в формате Server-Sent Events (text/event-stream).
    """
    rag_service = getattr(request.app.state, "rag_service", None)

    if not rag_service:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при загрузке RAG-модели"
        )

    async def event_generator():
        try:
            async for token in rag_service.ask_stream(
                query=body.query,
                session=session,
                top_k=body.top_k,
            ):
                yield f"data: {token}\n\n"

            yield "event: done\ndata: [DONE]\n\n"
        except Exception:
            yield "event: error\ndata: Внутренняя ошибка сервера\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
