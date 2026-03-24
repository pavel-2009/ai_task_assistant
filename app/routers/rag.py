"""
Роутер для запросов к RAG модели.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Request, status, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from typing import AsyncGenerator

from ..db import get_async_session
from ..ml.nlp.rag_service import RAGService


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
    q: str = Query(..., min_length=1, description="Вопрос пользователя"),
    top_k: int = Query(default=3, ge=1, le=10, description="Количество задач"),
    session: AsyncSession = Depends(get_async_session)
):
    """Стриминговый запрос к модели RAG"""
    
    
    rag_service: RAGService = getattr(request.app.state, "rag_service", None)

    if not rag_service:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при загрузке RAG-модели"
        )
        
    async def generate_stream() -> AsyncGenerator[str, None]:
        
        async for chunk in rag_service.ask_stream(q, session, top_k):
            
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )