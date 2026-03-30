"""
Сервис для управления локальной/облачной LLM моделью.
"""

import logging

from app.core import config

try:
    import ollama
except ImportError:  # pragma: no cover - зависит от окружения
    ollama = None


logger = logging.getLogger(__name__)


class LLMService:
    """Сервис для управления локальной/облачной LLM моделью."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.url = base_url or config.OLLAMA_BASE_URL
        self.client = ollama.AsyncClient(self.url) if ollama is not None else None
        self.model = model or config.OLLAMA_MODEL

    async def generate(self, prompt: str, system: str | None = None) -> str:
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
                messages=messages,
            )

            return response["message"]["content"]
        except Exception as exc:
            logger.error(f"Ошибка при генерации ответа: {exc}")
            return "Извините, произошла ошибка при обработке вашего запроса."

    async def generate_stream(self, prompt: str, system: str | None = None):
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
            stream=True,
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
            models = [model["name"] for model in response.get("models", [])]
            return self.model in models
        except Exception:
            return False

    async def warmup(self) -> None:
        """
        Гарантирует, что модель доступна в Ollama при старте.
        Если модели нет локально — пробует загрузить ее.
        """
        if self.client is None:
            logger.warning("Skip LLM warmup: package 'ollama' is not installed")
            return

        try:
            if await self.is_available():
                return

            logger.info("Ollama model '%s' not found locally. Pulling...", self.model)
            await self.client.pull(self.model)
            logger.info("Ollama model '%s' is ready", self.model)
        except Exception as exc:
            logger.warning("LLM warmup failed for model '%s': %s", self.model, exc)
