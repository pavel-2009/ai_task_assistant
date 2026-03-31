"""
Сервис для управления облачной LLM моделью (OpenAI-compatible API).
"""

import json
import logging
import time
from collections.abc import AsyncGenerator

import httpx

from app.core import config
from app.ml.metrics import MLMetricsCollector

logger = logging.getLogger(__name__)


class LLMService:
    """Сервис для управления облачной LLM моделью."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.metrics = MLMetricsCollector(self.__class__.__name__)
        load_start = time.perf_counter()
        self.url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.api_key = config.LLM_API_KEY
        self.timeout_seconds = config.LLM_TIMEOUT_SECONDS
        self.metrics.record_load_time(time.perf_counter() - load_start)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_messages(self, prompt: str, system: str | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """Запрос к LLM модели и получение полного ответа."""
        try:
            with self.metrics.time_inference():
                if not self.api_key:
                    logger.error("LLM API key is not configured")
                    self.metrics.record_error("MissingApiKey")
                    return "Извините, LLM сервис сейчас недоступен."

                payload = {
                    "model": self.model,
                    "messages": self._build_messages(prompt=prompt, system=system),
                }

                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        f"{self.url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                    response.raise_for_status()

                data = response.json()
                self.metrics.record_success()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            self.metrics.record_error(type(exc).__name__)
            logger.error("Ошибка при генерации ответа: %s", exc, exc_info=True)
            return "Извините, произошла ошибка при обработке вашего запроса."

    async def generate_stream(self, prompt: str, system: str | None = None) -> AsyncGenerator[str, None]:
        """Потоковая генерация (async generator): возвращает токены по мере генерации."""

        if not self.api_key:
            logger.error("LLM API key is not configured")
            return

        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt=prompt, system=system),
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{self.url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        chunk = line[6:].strip()
                        if chunk == "[DONE]":
                            break

                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue

                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
        except Exception as exc:
            logger.error("Ошибка при потоковой генерации: %s", exc, exc_info=True)

    async def is_available(self) -> bool:
        """Проверяет доступность удаленного LLM API."""

        if not self.api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.url}/models", headers=self._headers())
                response.raise_for_status()

            return True
        except Exception:
            return False

    async def warmup(self) -> None:
        """Проверяет доступность удаленной модели при старте."""

        if not self.api_key:
            logger.warning("Skip LLM warmup: LLM_API_KEY is not set")
            return

        if await self.is_available():
            logger.info("Cloud LLM API is available (model='%s')", self.model)
            return

        logger.error("LLM warmup failed: API is unavailable (model='%s')", self.model)
        raise RuntimeError("LLM API is unavailable")
