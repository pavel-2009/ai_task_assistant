"""
Роутер для обработки запросов, связанных с NLP (Natural Language Processing).
"""

from fastapi import APIRouter, HTTPException, Request, Body

import asyncio

from ..ml.embedding_service import EmbeddingService


router = APIRouter(
    prefix="/nlp",
    tags=["NLP"]
)


@router.post("/embedding", description="Получить эмбеддинг для текста")
async def get_embedding(request: Request, text: str | list[str] = Body(...)):
    """Получить эмбеддинг для текста"""
    
    if len(text) == 0:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    if isinstance(text, list) and any(len(t) == 0 for t in text):
        raise HTTPException(status_code=400, detail="Один из текстов в списке пустой")
    
    if isinstance(text, list) and len(text) > 10:
        raise HTTPException(status_code=400, detail="Слишком много текстов в списке. Максимум 10")
    
    if isinstance(text, str) and len(text) > 1000:
        raise HTTPException(status_code=400, detail="Текст слишком длинный. Максимум 1000 символов")
    
    if isinstance(text, list) and any(len(t) > 1000 for t in text):
        raise HTTPException(status_code=400, detail="Один из текстов в списке слишком длинный. Максимум 1000 символов")
    
    if isinstance(text, list) and any(not isinstance(t, str) for t in text):
        raise HTTPException(status_code=400, detail="Все элементы в списке должны быть строками")
    
    try:
        embedding_service: EmbeddingService = request.app.state.embedding_service
    except AttributeError:
        raise HTTPException(status_code=503, detail="EmbeddingService не инициализирован. Проверьте логи приложения")
    
    try:
        
        embedding = await asyncio.to_thread(embedding_service.encode, text)
        
        return {"embedding": embedding.tolist()} # Преобразуем numpy массив в список для JSON сериализации
    
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))