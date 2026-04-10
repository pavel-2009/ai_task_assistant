# AI Task Assistant

Сервис на FastAPI для управления задачами с JWT-аутентификацией, фоновой обработкой и демонстрационными ML-интеграциями для CV, NLP и рекомендаций.

## Что демонстрирует проект

- архитектуру FastAPI с разделением на routers/services;
- JWT-аутентификацию и связанные сценарии безопасности;
- асинхронную обработку через Celery + Redis;
- интеграцию ML-пайплайнов для CV и NLP;
- векторный поиск на индексах FAISS;
- мониторинг через health-check и метрики Prometheus.

## Архитектура

Границы слоёв и ответственность модулей описаны в `ARCHITECTURE.md`.

## Модули

- `auth` — аутентификация и работа с JWT;
- `tasks` — CRUD задач и жизненный цикл задач;
- `nlp` — эмбеддинги, семантический поиск, NER и RAG;
- `cv` — классификация, детекция, сегментация и эмбеддинги изображений;
- `recsys` — рекомендательные сервисы и обработка взаимодействий;
- `monitoring` — метрики, health-check и детекция дрейфа.

## Быстрый старт

### Docker

```bash
docker-compose up --build
```

`docker-compose` читает переменные окружения, поэтому для запуска через Docker можно либо экспортировать их в shell, либо положить в `.env`.

Рекомендуемые ключи `.env` для Docker:

```env
SECRET_KEY=your-secret-key-here
PYTHONUNBUFFERED=1
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=postgres
DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/postgres
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
LLM_API_KEY=your-actual-api-key-here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.3-8b-instruct:free
LLM_TIMEOUT_SECONDS=60
JWT_EXPIRE_MINUTES=60
JWT_ALGORITHM=HS256
USE_ONNX=false
CELERY_INIT_SERVICES_ON_STARTUP=false
MAX_IMAGE_SIZE_PX=1024
DEFAULT_TOP_K=5
FRAME_CACHE_SIZE=5
YOLO_CONF_THRESHOLD=0.35
YOLO_IOU_THRESHOLD=0.55
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD_SECONDS=60
METRICS_ENABLED=true
METRICS_PATH=/metrics
```

Доступные сервисы:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Метрики: `http://localhost:8000/metrics`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

### Локальный запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r prod_req.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Celery запускается отдельно:

```bash
celery -A app.celery_app worker -B --loglevel=info
```

## Тесты

Запуск всех тестов:

```bash
pytest
```

Запуск с покрытием:

```bash
pytest --cov=app
```

В репозитории есть модульные тесты для изолированной логики и интеграционные тесты для многошаговых сценариев.

## Мониторинг

- `/metrics` отдает метрики в формате Prometheus;
- API и Celery-инфраструктура инструментированы для наблюдаемости.

## Стиль кода

В проекте используются инструменты:

- `black`
- `ruff`
- `mypy`

## Полезные файлы

- `PORTFOLIO_REVIEW.md` — чек-лист портфолио-ревью;
- `requirements.txt` / `prod_req.txt` — зависимости;
- `docker-compose.yml` — локальная оркестрация API, PostgreSQL, Redis и Celery.

## Секреты CI

GitHub Actions CI может поднимать весь стек `docker-compose` из секретов репозитория. Workflow напрямую читает эти имена секретов:

- `SECRET_KEY`
- `PYTHONUNBUFFERED`
- `LLM_API_KEY`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `DATABASE_URL`
- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `JWT_EXPIRE_MINUTES`
- `JWT_ALGORITHM`
- `USE_ONNX`
- `CELERY_INIT_SERVICES_ON_STARTUP`
- `MAX_IMAGE_SIZE_PX`
- `DEFAULT_TOP_K`
- `FRAME_CACHE_SIZE`
- `YOLO_CONF_THRESHOLD`
- `YOLO_IOU_THRESHOLD`
- `RATE_LIMIT_REQUESTS`
- `RATE_LIMIT_PERIOD_SECONDS`
- `METRICS_ENABLED`
- `METRICS_PATH`
