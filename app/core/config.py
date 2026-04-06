from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    SECRET_KEY: str
    DATABASE_URL: str
    REDIS_URL: str
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    JWT_EXPIRE_MINUTES: int = 1
    JWT_ALGORITHM: str = "HS256"
    USE_ONNX: bool = False
    CELERY_INIT_SERVICES_ON_STARTUP: bool = False

    MAX_IMAGE_SIZE_PX: int = 1024
    DEFAULT_TOP_K: int = 5
    FRAME_CACHE_SIZE: int = 5
    YOLO_CONF_THRESHOLD: float = 0.35
    YOLO_IOU_THRESHOLD: float = 0.55
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD_SECONDS: int = 60
    METRICS_ENABLED: bool = True
    METRICS_PATH: str = "/metrics"

    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "meta-llama/llama-3.3-8b-instruct:free"
    LLM_API_KEY: str
    LLM_TIMEOUT_SECONDS: float = 60.0
    TEST_BASE_URL: str = "http://127.0.0.1:8000"

    INFERENCE_IDX_TO_CLASS: dict[int, str] = Field(default_factory=lambda: {0: "cat", 1: "dog", 2: "house"})
    
    USER_PROMPT: str = """Вот похожие задачи из системы:

%s

Вопрос пользователя: %s

Ответ:"""
    SYSTEM_PROMPT: str = """Ты — ассистент по управлению задачами в системе AI Task Assistant.
Отвечай ТОЛЬКО на основе информации из предоставленных задач.
Если ответа нет в задачах — скажи "У меня нет информации об этом в ваших задачах".
НЕ используй свои общие знания.
В конце ответа укажи ID задач, на которые опирался."""
    

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("SECRET_KEY must be set")
        return value

    @field_validator("JWT_EXPIRE_MINUTES")
    @classmethod
    def validate_jwt_expire_minutes(cls, value: int) -> int:
        if value <= 0 or value > 1440:
            raise ValueError("JWT_EXPIRE_MINUTES must be > 0 and <= 1440")
        return value


config = Settings()
