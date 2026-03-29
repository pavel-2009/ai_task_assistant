"""Rate limiting для API."""

from slowapi import Limiter
from app.core.config import config


limiter = Limiter(key_func=lambda: "global", default_limits=[f"{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_PERIOD_SECONDS}second"])