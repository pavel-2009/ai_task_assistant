"""
Сервис для управления локальной/облачной LLM моделью.
"""

import logging
import os

from dotenv import load_dotenv

try:
    import ollama
except ImportError:  # pragma: no cover - зависит от окружения
    ollama = None


logger = logging.getLogger(__name__)


load_dotenv()


class LLMService:
    """Сервис для управления локальной/облачной LLM моделью."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.url = os.getenv(
            "OLLAMA_BASE_URL",
            base_url if base_url else "http://localhost:11434"
        )
        
        self.client = ollama.AsyncClient(self.url) if ollama is not None else None
        
        self.model = os.getenv(
            "OLLAMA_MODEL",
            model if model else "llama3.2"
        )
        
        
    async def generate(self, prompt: str, system: str = None) -> str:
        """Запрос к LLM модели и получение полного ответа"""
        
        if self.client is None:
            logger.error("Ollama client is unavailable: package 'ollama' is not installed")
            return "Извините, LLM сервис сейчас недоступен."
        
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages
            )
            
            return response["message"]["content"]
        except Exception as e:
            # Логируем ошибку и возвращаем понятное сообщение
            logger.error(f"Ошибка при генерации ответа: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."
        
        
    async def generate_stream(self, prompt: str, system: str = None):
        """
        Потоковая генерация (async generator)
        Возвращает токены по мере генерации
        """
        
        if self.client is None:
            logger.error("Ollama client is unavailable: package 'ollama' is not installed")
            return

        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True
        )
        
        async for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    
    
    async def is_available(self) -> bool:
        """
        Проверяет доступность Ollama и наличие модели
        """
        if self.client is None:
            return False

        try:
            response = await self.client.list()
            
            models = [m["name"] for m in response.get("models", [])]
            
            return self.model in models
        
        except Exception:
            return False
