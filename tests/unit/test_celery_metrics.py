import pytest

from app.celery_metrics import _get_or_create_counter, track_celery_task


def test_decorator_success_execution():
    @track_celery_task('test_success_task')
    def job(x):
        return x + 1

    assert job(1) == 2


def test_decorator_error_execution():
    @track_celery_task('test_error_task')
    def job():
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError):
        job()


def test_get_existing_metric():
    first = _get_or_create_counter('test_celery_counter_total', 'Test counter', ['x'])
    second = _get_or_create_counter('test_celery_counter_total', 'Test counter', ['x'])
    assert first is second
