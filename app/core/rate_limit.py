"""Rate limiting для API."""

from slowapi import Limiter
from slowapi.util import get_remote_address
from app.core.config import config


limiter = Limiter(key_func=get_remote_address, default_limits=[f"{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_PERIOD_SECONDS}second"])