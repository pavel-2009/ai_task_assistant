# AI Task Assistant

Production-style backend pet project on **FastAPI + Celery + Redis + SQLAlchemy** with emphasis on demonstrating multiple technology tracks in one codebase:

- REST API with JWT auth.
- Async DB access and migrations (Alembic).
- Background processing with Celery.
- NLP pipeline (embeddings, NER, semantic search, RAG).
- CV pipeline (classification, detection, segmentation, image embeddings).
- Recommender components (content-based + collaborative filtering internals).
- Basic monitoring/health/rate limiting.

> **Project goal:** portfolio demonstration of architecture and integrations. Business domain is task management.

---

## 1) Implemented functionality (current state)

### Core API
- User registration and login with JWT token issuance.
- Task CRUD.
- Task like interaction endpoint.
- Background enrichment of task tags/embeddings after create/update.

### Avatars / CV
- Upload avatar image for task.
- Async Celery jobs for:
  - avatar class prediction,
  - object detection on avatar,
  - image segmentation.
- Endpoints for polling Celery task statuses/results.

### NLP / RAG
- Build embedding for one text or a batch.
- Semantic indexing and search.
- NER-based task tagging.
- RAG ask endpoint + streaming SSE endpoint.
- Manual RAG reindex trigger.

### Monitoring / reliability
- `/ping` health endpoint with model/service readiness.
- Drift report and drift history endpoints.
- Request rate limiting on auth endpoints.
- Prometheus metrics integration hook.

### Recsys notes
- Recsys services and tasks are present in code and used internally by background jobs.
- `recsys` router file exists, but this router is not included in `app.main` by default.

---

## 2) Tech stack

- **Backend:** FastAPI, Uvicorn.
- **Background jobs:** Celery + Redis broker/backend.
- **DB layer:** SQLAlchemy (async), Alembic.
- **Validation/settings:** Pydantic, pydantic-settings.
- **NLP/LLM:** sentence-transformers style embedding flow, NER service, external LLM API for RAG.
- **CV/ML:** PyTorch + YOLO/ONNX utilities, segmentation/embedding helpers.
- **Tests:** pytest (unit + integration scenarios in repository).
- **Infra:** Docker, docker-compose.

---

## 3) Project structure

```text
app/
  core/                 # settings, security, metrics, rate limits, deps
  routers/              # API routers
  ml/                   # NLP/CV/Recsys/Monitoring modules
  schemas/              # Pydantic schemas
  db.py                 # DB engine/session wiring
  db_models.py          # SQLAlchemy models
  services.py           # global service registry + init
  main.py               # FastAPI app entrypoint + lifespan
alembic/                # DB migrations
tests/                  # unit and integration tests
Dockerfile
docker-compose.yml
prod_req.txt
requirements.txt
```

---

## 4) API routes (high-level)

### Auth (`/auth`)
- `POST /auth/register`
- `POST /auth/login`

### Tasks (`/tasks`)
- `GET /tasks/`
- `POST /tasks/`
- `GET /tasks/{task_id}`
- `PUT /tasks/{task_id}`
- `DELETE /tasks/{task_id}`
- `GET /tasks/{task_id}/tags_status`
- `POST /tasks/{task_id}/like`

### Avatars/CV (`/tasks/...`)
- `POST /tasks/{task_id}/avatar`
- `POST /tasks/{task_id}/predict/submit`
- `GET /tasks/{task_id}/predict/status/{celery_task_id}`
- `POST /tasks/{task_id}/detect/submit`
- `GET /tasks/{task_id}/detect/status/{celery_task_id}`
- `POST /tasks/{task_id}/segment/submit`
- `GET /tasks/{task_id}/segment/status/{celery_task_id}`

### NLP (`/nlp`)
- `POST /nlp/embedding`
- `POST /nlp/search`
- `POST /nlp/index`
- `POST /nlp/tag-task`

### RAG (`/rag`)
- `POST /rag/reindex`
- `POST /rag/ask`
- `POST /rag/ask/stream`

### Streaming (`/ws`)
- `WS /ws/detect`

### Monitoring (`/monitoring`)
- `GET /monitoring/drift/report`
- `GET /monitoring/drift/history`

### Health
- `GET /ping`

---

## 5) Configuration (.env)

The app uses centralized settings from `app/core/config.py`.

### Required
- `SECRET_KEY`
- `DATABASE_URL`
- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `LLM_API_KEY`

### Important optional
- `JWT_EXPIRE_MINUTES` (default `1`)
- `JWT_ALGORITHM` (default `HS256`)
- `USE_ONNX` (default `False`)
- `CELERY_INIT_SERVICES_ON_STARTUP` (default `False`)
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `DEFAULT_TOP_K`
- `METRICS_ENABLED`
- `METRICS_PATH`

Example `.env`:

```env
SECRET_KEY=super-secret-key
DATABASE_URL=sqlite+aiosqlite:///./data/app.db
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
LLM_API_KEY=your_api_key

JWT_EXPIRE_MINUTES=60
JWT_ALGORITHM=HS256
USE_ONNX=false
CELERY_INIT_SERVICES_ON_STARTUP=false
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.3-8b-instruct:free
```

---

## 6) Run with Docker (recommended)

```bash
docker-compose up --build
```

After startup:
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Ping: `http://localhost:8000/ping`

`docker-compose.yml` starts three services:
- `redis`
- `api` (Uvicorn)
- `celery` (worker + beat)

---

## 7) Local run (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r prod_req.txt
```

Run API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run Celery worker/beat in another terminal:

```bash
celery -A app.celery_app worker -B --loglevel=info
```

Optional migrations:

```bash
alembic upgrade head
```

---

## 8) Tests

Repository includes both unit and integration tests under `tests/`.

Typical commands:

```bash
pytest -q
pytest tests/unit -q
pytest tests/integration -q
```

Integration tests expect running app/infra and proper env configuration.

---

## 9) Portfolio highlights

- One repository showing **async API + background workers + ML/NLP/CV + monitoring**.
- Explicit separation of concerns (`routers`, `services`, `ml`, `core`, `schemas`).
- Centralized typed config and startup lifecycle orchestration.
- Realistic infra wiring (Docker, Redis, Celery, Alembic, health checks).

---

## 10) Known constraints

- Some ML features depend on heavyweight external models/resources.
- Full integration scenarios require running Redis/Celery and valid LLM API key.
- Functionality breadth is prioritized for demo/portfolio coverage.
