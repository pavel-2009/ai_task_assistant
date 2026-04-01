"""Unified common/shared response schemas for API."""

from pydantic import BaseModel, Field
from typing import Any, Optional, Literal


class ErrorResponse(BaseModel):
    """Unified error response with code and detail."""

    code: str = Field(description="Error code for programmatic handling (e.g., 'RESOURCE_NOT_FOUND', 'INVALID_INPUT')")
    detail: str = Field(description="Human-readable error message")


class SuccessMessageResponse(BaseModel):
    """Generic success response with a message."""

    message: str = Field(description="Success message")


class MessageDetailResponse(BaseModel):
    """Response with message and detail fields."""

    message: str = Field(description="Message")
    detail: str = Field(description="Additional details")


class CeleryTaskResponse(BaseModel):
    """Response for async Celery task submission."""

    message: str = Field(description="Message about task submission")
    celery_task_id: str = Field(description="Celery task ID for status tracking")


class CeleryTaskStatusBase(BaseModel):
    """Base class for Celery task status responses."""

    status: str = Field(description="Task status (PENDING, STARTED, SUCCESS, FAILURE)")
    message: Optional[str] = Field(default=None, description="Status message")


class CeleryTaskStatusResponse(CeleryTaskStatusBase):
    """Response for checking Celery task status."""

    result: Optional[Any] = Field(default=None, description="Task result (for SUCCESS status)")


class FileUploadResponse(BaseModel):
    """Response for file upload operations."""

    filepath: str = Field(description="Path to the uploaded file")
    filename: str = Field(description="Name of the uploaded file")


class EmbeddingResponse(BaseModel):
    """Response containing embeddings."""

    embedding: list[list[float]] | list[float] = Field(description="Embedding vectors")


class SearchResultItem(BaseModel):
    """A single search result item."""

    text_id: str = Field(description="Text/document ID in vector index")
    task_id: Optional[int] = Field(default=None, description="Task ID when text belongs to task")
    text: Optional[str] = Field(default=None, description="Indexed source text")
    title: Optional[str] = Field(default=None, description="Task title")
    description: Optional[str] = Field(default=None, description="Task description")
    similarity: Optional[float] = Field(default=None, description="Search relevance score")
    score: Optional[float] = Field(default=None, description="Alias of similarity score")


class SearchResults(BaseModel):
    """Search results collection."""

    results: list[SearchResultItem] = Field(description="List of search results")
    total: int = Field(default=0, description="Total number of results")


class TagsResponse(BaseModel):
    """Response containing tags."""

    tags: Optional[list[str]] = Field(default=None, description="List of tags")


class TaskStatusResponse(BaseModel):
    """Response indicating task processing status."""

    tags: Optional[str] = Field(default=None, description="Processed tags")
    is_processing: bool = Field(description="Whether tags are still being processed")


class TokenResponse(BaseModel):
    """Response containing authentication token."""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(description="Token type, typically 'bearer'")


class HealthComponentStatus(BaseModel):
    """Health status of a single component."""

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
    """Response for health-check endpoint."""

    status: Literal["ok", "degraded"] = Field(description="Overall status ('ok' if all components ready, 'degraded' otherwise)")
    models_ready: bool = Field(description="Whether all models are ready")
    models: dict[str, HealthComponentStatus] = Field(description="Status of each component")


class StreamingChunkResponse(BaseModel):
    """Response chunk for streaming endpoints."""

    chunk: str = Field(description="Text chunk from the stream")
    is_done: bool = Field(default=False, description="Whether this is the final chunk")


class FileDownloadResponse(BaseModel):
    """Metadata for file download response."""

    filename: str = Field(description="Name of the file")
    size: Optional[int] = Field(default=None, description="File size in bytes")


class DriftReportResponse(BaseModel):
    """Response containing drift detection report."""

    status: str = Field(description="Status of the drift detector")
    detail: Optional[str] = Field(default=None, description="Additional details")


class DriftHistoryResponse(BaseModel):
    """Response containing drift history."""

    history: dict = Field(default_factory=dict, description="Historical drift alerts")
    count: int = Field(description="Number of drift alerts")
    status: str = Field(description="Status of history retrieval")
    error: Optional[str] = Field(default=None, description="Error message if any")


class AskResponse(BaseModel):
    """Response from RAG ask endpoint."""

    answer: str = Field(description="Answer from RAG model")
    sources: Optional[list[dict]] = Field(default=None, description="Source documents used")
    confidence: Optional[float] = Field(default=None, description="Average similarity confidence")
    cached: Optional[bool] = Field(default=None, description="Whether result was loaded from cache")


class NLPTagTaskResponse(BaseModel):
    """Response for NLP tag-task endpoint."""

    tags: list[str] = Field(description="Extracted tags/entities from text")


class IndexResponse(BaseModel):
    """Response for text indexing."""

    detail: str = Field(description="Indexing status message")
