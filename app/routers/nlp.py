"""
Роутер для обработки запросов, связанных с NLP (Natural Language Processing).
"""

from fastapi import APIRouter, HTTPException, Request, Body, status

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
        
        match text:
            case str():
                embedding = await asyncio.to_thread(embedding_service.encode_one, text)
            case list():
                embedding = await asyncio.to_thread(embedding_service.encode_batch, text)
            case _:
                raise HTTPException(status_code=400, detail="Неверный формат данных. Ожидается строка или список строк")
        
        return {"embedding": embedding.tolist()} # Преобразуем numpy массив в список для JSON сериализации
    
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/search", description="Искать документы, наиболее похожие на запрос", status_code=status.HTTP_200_OK)
async def search(
    request: Request,
    query: str = Body(..., embed=True, description="Текст запроса для поиска"),
    top_k: int = Body(5, embed=True, description="Количество результатов для возврата")
):
    """Поиск документов, наиболее похожих на запрос"""
    
    if len(query) == 0:
        raise HTTPException(status_code=400, detail="Запрос не может быть пустым")
    
    if top_k <= 0 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k должен быть в диапазоне от 1 до 20")
    
    try:
        embedding_service: EmbeddingService = request.app.state.embedding_service
    except AttributeError:
        raise HTTPException(status_code=503, detail="EmbeddingService не инициализирован. Проверьте логи приложения")
    
    try:
        semantic_search_service = request.app.state.semantic_search_service
    except AttributeError:
        raise HTTPException(status_code=503, detail="SemanticSearchService не инициализирован. Проверьте логи приложения")
    
    try:
        results = await asyncio.to_thread(semantic_search_service.search, query, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/index", description="Индексировать текст, добавляя его эмбеддинг в базу данных", status_code=status.HTTP_200_OK)
async def index(
    request: Request,
    text: str = Body(..., embed=True, description=""),
):
    """Индексировать текст, добавляя его эмбеддинг в базу данных"""
    
    if len(text) == 0:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    try:
        embedding_service: EmbeddingService = request.app.state.embedding_service
    except AttributeError:
        raise HTTPException(status_code=503, detail="EmbeddingService не инициализирован. Проверьте логи приложения")
    
    try:
        semantic_search_service = request.app.state.semantic_search_service
    except AttributeError:
        raise HTTPException(status_code=503, detail="SemanticSearchService не инициализирован. Проверьте логи приложения")
    
    try:
        await asyncio.to_thread(semantic_search_service.index, text)
        return {"detail": "Текст успешно индексирован"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))