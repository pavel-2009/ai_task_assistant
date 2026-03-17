"""
Роутер для обработки запросов, связанных с NLP (Natural Language Processing).
"""

from fastapi import APIRouter, HTTPException, Request

from ..ml.embedding_service import EmbeddingService


router = APIRouter(
    prefix="/nlp",
    tags=["NLP"]
)


@router.post("/embedding", description="Получить эмбеддинг для текста")
async def get_embedding(text: str, request: Request):
    """Получить эмбеддинг для текста"""
    
    try:
        embedding_service: EmbeddingService = request.app.state.embedding_service
    except AttributeError:
        raise HTTPException(status_code=503, detail="EmbeddingService не инициализирован. Проверьте логи приложения")
    
    try:
        embedding = embedding_service.encode(text)
        
        return {"embedding": embedding.tolist()} # Преобразуем numpy массив в список для JSON сериализации
    
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))