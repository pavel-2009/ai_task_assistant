"""Service package with business services and shared ML service registry."""

from .auth_service import AuthService
from .registry import (
    default_inference_checkpoint_path,
    ensure_services_initialized,
    get_collaborative_filtering_recommender,
    get_content_based_recommender,
    get_drift_detector,
    get_embedding,
    get_image_embedding,
    get_inference,
    get_llm,
    get_ner,
    get_rag,
    get_recsys_vector_db,
    get_redis,
    get_segmentation,
    get_semantic_search,
    get_service,
    get_vector_db,
    get_yolo,
    init_services,
)
from .task_service import TaskService
from .user_service import UserService

__all__ = [
    "AuthService",
    "TaskService",
    "UserService",
    "default_inference_checkpoint_path",
    "ensure_services_initialized",
    "get_collaborative_filtering_recommender",
    "get_content_based_recommender",
    "get_drift_detector",
    "get_embedding",
    "get_image_embedding",
    "get_inference",
    "get_llm",
    "get_ner",
    "get_rag",
    "get_recsys_vector_db",
    "get_redis",
    "get_segmentation",
    "get_semantic_search",
    "get_service",
    "get_vector_db",
    "get_yolo",
    "init_services",
]
