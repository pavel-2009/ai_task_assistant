# AI Task Assistant

FastAPI service for task management with JWT auth, background processing, and demo ML integrations across CV, NLP, and recommendations.

## What this project demonstrates

- FastAPI architecture with routers/services separation
- JWT authentication and security flows
- Async processing with Celery + Redis
- ML integration across CV and NLP pipelines
- Vector search with FAISS-backed indexes
- Monitoring via health checks and Prometheus metrics

## Architecture

See `ARCHITECTURE.md` for layer boundaries and module responsibilities.

## Modules

- `auth` - authentication and JWT handling
- `tasks` - task CRUD and lifecycle flows
- `nlp` - embeddings, semantic search, NER, and RAG
- `cv` - classification, detection, segmentation, and image embeddings
- `recsys` - recommendation services and interaction processing
- `monitoring` - metrics, health checks, and drift detection

## Quick Start

### Docker

```bash
docker-compose up --build
```

Compose reads environment variables, so for Docker startup you can either export them in your shell or place them in `.env`.

Recommended Docker `.env` keys:

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

Available services:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Metrics: `http://localhost:8000/metrics`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

### Local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r prod_req.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run Celery separately:

```bash
celery -A app.celery_app worker -B --loglevel=info
```

## Tests

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app
```

The repository includes unit tests for isolated logic and integration tests for multi-component flows.

## Monitoring

- `/metrics` exposes Prometheus-compatible metrics
- API and Celery activity are instrumented for observability

## Code Style

Project tooling includes:

- `black`
- `ruff`
- `mypy`

## Useful Files

- `PORTFOLIO_REVIEW.md` - portfolio review checklist
- `requirements.txt` / `prod_req.txt` - dependencies
- `docker-compose.yml` - local orchestration for API, PostgreSQL, Redis, and Celery

## CI Secrets

GitHub Actions CI can start the whole `docker-compose` stack from repository secrets. The workflow reads these secret names directly:

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
