from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SECRET_KEY: str
    DATABASE_URL: str = 'sqlite+aiosqlite:///./test.db'
    REDIS_URL: str
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    JWT_EXPIRE_MINUTES: int = 60
    JWT_ALGORITHM: str = "HS256"
    USE_ONNX: bool = False

    MAX_IMAGE_SIZE_PX: int = 1024
    DEFAULT_TOP_K: int = 5
    FRAME_CACHE_SIZE: int = 5
    YOLO_CONF_THRESHOLD: float = 0.35
    YOLO_IOU_THRESHOLD: float = 0.55
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD_SECONDS: int = 60

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

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
