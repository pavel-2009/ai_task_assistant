from starlette.requests import Request

from app.core.config import config
from app.core.rate_limit import limiter
from slowapi.util import get_remote_address


def test_create_limiter_with_defaults():
    assert limiter.default_limits == [f'{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_PERIOD_SECONDS}second']


def test_get_remote_address():
    scope = {
        'type': 'http',
        'method': 'GET',
        'path': '/',
        'headers': [],
        'client': ('127.0.0.1', 5555),
        'scheme': 'http',
        'server': ('testserver', 80),
    }
    request = Request(scope)
    assert get_remote_address(request) == '127.0.0.1'
