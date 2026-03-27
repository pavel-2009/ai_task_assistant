"""
Роутер для запросов к RAG модели.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_async_session
from ..error_handlers import AppError
from ..schemas import AskRequest


router = APIRouter(
    prefix="/rag",
    tags=['RAG']
)



@router.post("/ask")
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
        return result
    except Exception as exc:
        raise AppError("Ошибка при обработке RAG запроса", status_code=500) from exc


@router.post("/ask/stream")
async def ask_stream(
    request: Request,
    body: AskRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Потоковый SSE-запрос к RAG модели.
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
