"""Единые общие схемы ответов API."""

from pydantic import BaseModel, Field
from typing import Any, Optional, Literal


class ErrorResponse(BaseModel):
    """Единый ответ об ошибке с кодом и описанием."""

    code: str = Field(description="Error code for programmatic handling (e.g., 'RESOURCE_NOT_FOUND', 'INVALID_INPUT')")
    detail: str = Field(description="Human-readable error message")


class SuccessMessageResponse(BaseModel):
    """Универсальный успешный ответ с сообщением."""

    message: str = Field(description="Success message")


class MessageDetailResponse(BaseModel):
    """Ответ с полями message и detail."""

    message: str = Field(description="Message")
    detail: str = Field(description="Additional details")


class CeleryTaskResponse(BaseModel):
    """Ответ при отправке асинхронной задачи Celery."""

    message: str = Field(description="Message about task submission")
    celery_task_id: str = Field(description="Celery task ID for status tracking")


class CeleryTaskStatusBase(BaseModel):
    """Базовый класс ответов о статусе задачи Celery."""

    status: str = Field(description="Task status (PENDING, STARTED, SUCCESS, FAILURE)")
    message: Optional[str] = Field(default=None, description="Status message")


class CeleryTaskStatusResponse(CeleryTaskStatusBase):
    """Ответ для проверки статуса задачи Celery."""

    result: Optional[Any] = Field(default=None, description="Task result (for SUCCESS status)")


class FileUploadResponse(BaseModel):
    """Ответ для операций загрузки файлов."""

    filepath: str = Field(description="Path to the uploaded file")
    filename: str = Field(description="Name of the uploaded file")


class EmbeddingResponse(BaseModel):
    """Ответ, содержащий эмбеддинги."""

    embedding: list[list[float]] | list[float] = Field(description="Embedding vectors")


class SearchResultItem(BaseModel):
    """Один элемент результата поиска."""

    text_id: str = Field(description="Text/document ID in vector index")
    task_id: Optional[int] = Field(default=None, description="Task ID when text belongs to task")
    text: Optional[str] = Field(default=None, description="Indexed source text")
    title: Optional[str] = Field(default=None, description="Task title")
    description: Optional[str] = Field(default=None, description="Task description")
    similarity: Optional[float] = Field(default=None, description="Search relevance score")
    score: Optional[float] = Field(default=None, description="Alias of similarity score")


class SearchResults(BaseModel):
    """Коллекция результатов поиска."""

    results: list[SearchResultItem] = Field(description="List of search results")
    total: int = Field(default=0, description="Total number of results")


class TagsResponse(BaseModel):
    """Ответ, содержащий теги."""

    tags: Optional[list[str]] = Field(default=None, description="List of tags")


class TaskStatusResponse(BaseModel):
    """Ответ со статусом обработки задачи."""

    tags: Optional[str] = Field(default=None, description="Processed tags")
    is_processing: bool = Field(description="Whether tags are still being processed")


class TokenResponse(BaseModel):
    """Ответ, содержащий токен аутентификации."""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(description="Token type, typically 'bearer'")


class HealthComponentStatus(BaseModel):
    """Статус здоровья отдельного компонента."""

    ready: bool = Field(description="Component readiness status")
    model: Optional[str] = Field(default=None, description="Model name or version")
    dimension: Optional[int] = Field(default=None, description="Embedding dimension")
    indexed_items: Optional[int] = Field(default=None, description="Number of indexed items")
    cache_prefix: Optional[str] = Field(default=None, description="Cache prefix")
    url: Optional[str] = Field(default=None, description="Service URL")
    llm_model: Optional[str] = Field(default=None, description="LLM model name")
    drift_detected: Optional[bool] = Field(default=None, description="Drift detection status")
    drift_score: Optional[float] = Field(default=None, description="Drift score")


class PingResponse(BaseModel):
    """Ответ для endpoint проверки состояния."""

    status: Literal["ok", "degraded"] = Field(description="Overall status ('ok' if all components ready, 'degraded' otherwise)")
    models_ready: bool = Field(description="Whether all models are ready")
    models: dict[str, HealthComponentStatus] = Field(description="Status of each component")


class StreamingChunkResponse(BaseModel):
    """Фрагмент ответа для потоковых endpoint-ов."""

    chunk: str = Field(description="Text chunk from the stream")
    is_done: bool = Field(default=False, description="Whether this is the final chunk")


class FileDownloadResponse(BaseModel):
    """Метаданные ответа на скачивание файла."""

    filename: str = Field(description="Name of the file")
    size: Optional[int] = Field(default=None, description="File size in bytes")


class DriftReportResponse(BaseModel):
    """Ответ, содержащий отчёт о детекции дрейфа."""

    status: str = Field(description="Status of the drift detector")
    detail: Optional[str] = Field(default=None, description="Additional details")


class DriftHistoryResponse(BaseModel):
    """Ответ, содержащий историю дрейфа."""

    history: dict = Field(default_factory=dict, description="Historical drift alerts")
    count: int = Field(description="Number of drift alerts")
    status: str = Field(description="Status of history retrieval")
    error: Optional[str] = Field(default=None, description="Error message if any")


class AskResponse(BaseModel):
    """Ответ endpoint-а RAG ask."""

    answer: str = Field(description="Answer from RAG model")
    sources: Optional[list[dict]] = Field(default=None, description="Source documents used")
    confidence: Optional[float] = Field(default=None, description="Average similarity confidence")
    cached: Optional[bool] = Field(default=None, description="Whether result was loaded from cache")


class NLPTagTaskResponse(BaseModel):
    """Ответ для NLP endpoint-а tag-task."""

    tags: list[str] = Field(description="Extracted tags/entities from text")


class IndexResponse(BaseModel):
    """Ответ для индексации текста."""

    detail: str = Field(description="Indexing status message")
