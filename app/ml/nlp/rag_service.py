"""
Сервис для Retrieval-Augmented Generation (RAG)
"""

import hashlib
import json
from typing import TYPE_CHECKING, Any, Dict, List, AsyncGenerator

if TYPE_CHECKING:
    from redis.asyncio import Redis as AsyncRedis
    from sqlalchemy.ext.asyncio import AsyncSession
    from .llm_service import LLMService
    from .semantic_search_service import SemanticSearchService


class RAGService:
    """Сервис для Retrieval-Augmented Generation (RAG)"""
    
    def __init__(
        self,
        llm_service: "LLMService",
        semantic_search_service: "SemanticSearchService",
        redis: "AsyncRedis | None" = None,
    ):
        self.llm_service = llm_service
        self.semantic_search_service = semantic_search_service
        self.redis_client = redis
        
        
    def _format_tasks(self, tasks: List[dict]) -> str:
        """Форматирует найденные задачи для промпта."""
        result = []
        for i, task in enumerate(tasks, 1):
            result.append(
                f"""--- ЗАДАЧА {i} (похожесть: {task['similarity']:.2f}) ---
Название: {task.get('title', 'Без названия')}
Описание: {task.get('description', 'Нет описания')}
Технологии: {task.get('tags', 'не указаны')}
ID: {task.get('task_id')}"""
            )
        return "\n".join(result)
    
    
    def _build_sources(self, tasks: List[dict]) -> List[dict]:
        """Формирует список источников для ответа."""
        return [
            {
                "task_id": task.get("task_id"),
                "title": task.get("title"),
                "similarity": task.get("similarity"),
            }
            for task in tasks
        ]
        
        
    def _calculate_confidence(self, tasks: List[dict]) -> float:
        """Возвращает среднюю похожесть найденных задач."""
        if not tasks:
            return 0
        return sum(task.get("similarity", 0) for task in tasks) / len(tasks)


    @staticmethod
    def _get_cache_key(query: str, top_k: int) -> str:
        """Генерирует ключ для кэша."""
        return f"rag:{hashlib.md5(query.encode()).hexdigest()}:{top_k}"
    
    
    def _chunk_text(self, text: str, chunk_size: int = 5) -> List[str]:
        """Разбивает текст на чанки по словам для имитации стриминга"""
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i + chunk_size]))
            
        return chunks if chunks else [text]
    
    
    async def ask(
        self,
        query: str,
        session: "AsyncSession",
        top_k: int = 3,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Получение ответа на вопрос с помощью RAG"""
        
        cache_key = self._get_cache_key(query, top_k)

        if use_cache and self.redis_client:
            cached_response = await self.redis_client.get(cache_key)

            if cached_response:
                if isinstance(cached_response, bytes):
                    cached_response = cached_response.decode("utf-8")
                response = json.loads(cached_response)
                response["cached"] = True
                return response
            
        tasks = await self.semantic_search_service.search(query, session, top_k)
        
        if not tasks:
            return {
                "message": (
                    "У меня нет информации об этом в ваших задачах. "
                    "Попробуйте переформулировать вопрос или создать задачу по этой теме."
                ),
                "sources": [],
                "confidence": 0,
                "cached": False,
            }
            
        formatted_tasks = self._format_tasks(tasks)
        sources = self._build_sources(tasks)
        confidence = self._calculate_confidence(tasks)
        
        system_prompt = """Ты — ассистент по управлению задачами в системе AI Task Assistant.
Отвечай ТОЛЬКО на основе информации из предоставленных задач.
Если ответа нет в задачах — скажи "У меня нет информации об этом в ваших задачах".
НЕ используй свои общие знания.
В конце ответа укажи ID задач, на которые опирался."""

        user_prompt = f"""Вот похожие задачи из системы:

{formatted_tasks}

Вопрос пользователя: {query}

Ответ:"""

        answer = await self.llm_service.generate(
            prompt=user_prompt,
            system=system_prompt
        )
        
        response = {
            "message": answer,
            "sources": sources,
            "confidence": confidence,
            "cached": False,
        }
        
        # Кэшируем
        if use_cache and self.redis_client:
            await self.redis_client.setex(
                cache_key,
                300,
                json.dumps(response, ensure_ascii=False)
            )
            
        return response
    
    
    async def ask_stream(
        self,
        query: str,
        session: "AsyncSession",
        top_k: int = 3,
        use_cache: bool = True
    ) -> AsyncGenerator[dict, None]:
        """Получение стримингового ответа на вопрос от RAG"""
        
        cache_key = self._get_cache_key(query, top_k)
    
        # Проверяем кэш
        if use_cache and self.redis_client:
            cached = await self.redis_client.get(cache_key)
            if cached:
                # Воспроизводим закэшированный ответ как стрим
                response = json.loads(cached)
                # Отдаем метаданные
                yield {
                    "type": "meta",
                    "sources": response["sources"],
                    "confidence": response["confidence"],
                    "cached": True,
                }
                # Отдаем токены (разбиваем ответ на чанки)
                for token in self._chunk_text(response["answer"]):
                    yield {"type": "token", "content": token}
                yield {"type": "done"}
                return
        
        tasks = await self.semantic_search_service.search(query, session, top_k)
        
        if not tasks:
            yield {
                "type": "error",
                "message": (
                    "У меня нет информации об этом в ваших задачах. "
                    "Попробуйте переформулировать вопрос или создать задачу по этой теме."
                ),
            }
            yield {
                "type": "done"
            }
            return

        formatted_tasks = self._format_tasks(tasks)
        sources = self._build_sources(tasks)
        confidence = self._calculate_confidence(tasks)
        
        system_prompt = """Ты — ассистент по управлению задачами в системе AI Task Assistant.
Отвечай ТОЛЬКО на основе информации из предоставленных задач.
Если ответа нет в задачах — скажи "У меня нет информации об этом в ваших задачах".
НЕ используй свои общие знания.
В конце ответа укажи ID задач, на которые опирался."""

        user_prompt = f"""Вот похожие задачи из системы:

{formatted_tasks}

Вопрос пользователя: {query}

Ответ:"""

        stream = await self.llm_service.generate_stream(prompt=user_prompt, system=system_prompt)
        
        try:
            yield {
                "type": "meta",
                "sources": sources,
                "confidence": confidence,
                "cached": False,
            }

            async for chunk in stream:
                yield {
                    "type": "token",
                    "content": str(chunk)
                }
                
        except Exception as e:
            
            yield {"type": "error", "message": str(e)}
            
        yield {
            "type": "done"
        }
