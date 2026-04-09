"""Rate limiting для API."""

from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import config


default_limits = [f"{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_PERIOD_SECONDS}second"]
limiter = Limiter(key_func=get_remote_address, default_limits=default_limits)
# Совместимость с тестами/старыми версиями, где ожидался публичный атрибут default_limits.
if not hasattr(limiter, "default_limits"):
    limiter.default_limits = default_limits
