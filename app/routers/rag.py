"""
Роутер для запросов к RAG модели.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_async_session


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
