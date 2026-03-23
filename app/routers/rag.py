"""
Роутер для запросов к RAG модели.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request, Body, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import json

from ...app.db import get_async_session


router = APIRouter(
    prefix="/rag",
    tags=['RAG']
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    use_cache: bool = True


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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/ask/stream")
async def ask_stream(
    request: Request,
    query: str = Query(..., min_length=1),
    top_k: int = Query(3, ge=1, le=10),
):
    """
    Стриминговый ответ от RAG модели (SSE).
    """

    rag_service = getattr(request.app.state, "rag_service", None)

    if not rag_service:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG сервис не инициализирован"
        )

    async def event_generator():
        try:
            
            sources = await rag_service.semantic_search.search(query, top_k)

            yield f"data: {json.dumps({'sources': sources})}\n\n"

            prompt = await rag_service.build_prompt(query, sources)

            async for token in rag_service.llm.generate_stream(prompt):
                yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )