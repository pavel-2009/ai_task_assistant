# AI Task Assistant

FastAPI‑сервис для управления задачами с авторизацией, загрузкой изображений, NLP/RAG, CV/ML‑модулями и фоновыми задачами через Celery + Redis.

---

## Что есть в проекте

- Авторизация (JWT), пользователи и сессии.
- CRUD задач.
- Загрузка/обработка изображений для задач.
- NLP: эмбеддинги, NER, semantic search, RAG.
- CV: классификация, детекция, сегментация, image embeddings.
- RecSys: content-based и collaborative filtering.
- Мониторинг и health-check endpoint.
- Celery-задачи для CV/NLP/RecSys.

### Важно: 1000 классов в классификации

В `app/ml/common/config.py` параметр `num_classes` вычисляется из `ResNet18_Weights.IMAGENET1K_V1`, что соответствует **1000 классам ImageNet (IMAGENET1K)**.

---

## Актуальная структура проекта

```text
.
├── app/
│   ├── core/
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   ├── metrics.py
│   │   ├── rate_limit.py
│   │   └── security.py
│   ├── ml/
│   │   ├── common/
│   │   │   └── config.py
│   │   ├── cv/
│   │   │   ├── classification/
│   │   │   │   ├── datasets.py
│   │   │   │   ├── inference_service.py
│   │   │   │   ├── models_nn.py
│   │   │   │   └── train_loop.py
│   │   │   ├── detection/
│   │   │   │   ├── yolo_onnx/
│   │   │   │   │   ├── config.py
│   │   │   │   │   ├── postprocessing.py
│   │   │   │   │   ├── preprocessing.py
│   │   │   │   │   ├── service.py
│   │   │   │   │   └── utils.py
│   │   │   │   ├── train_yolo.py
│   │   │   │   ├── yolo_onnx_service.py
│   │   │   │   └── yolo_service.py
│   │   │   ├── embedding/
│   │   │   │   └── image_embedding_service.py
│   │   │   ├── segmentation/
│   │   │   │   └── segmentation_service.py
│   │   │   └── tasks.py
│   │   ├── monitoring/
│   │   │   └── drift_detector.py
│   │   ├── nlp/
│   │   │   ├── embedding_service.py
│   │   │   ├── llm_service.py
│   │   │   ├── ner_service.py
│   │   │   ├── rag_service.py
│   │   │   ├── semantic_search_service.py
│   │   │   ├── tasks.py
│   │   │   └── vector_db.py
│   │   ├── recsys/
│   │   │   ├── collaborative_filtering.py
│   │   │   ├── content_based.py
│   │   │   ├── tasks.py
│   │   │   └── vector_db/recsys_vector_db.py
│   │   └── metrics.py
│   ├── routers/
│   │   ├── auth.py
│   │   ├── avatars.py
│   │   ├── monitoring.py
│   │   ├── nlp.py
│   │   ├── rag.py
│   │   ├── recsys.py
│   │   ├── streaming.py
│   │   └── tasks.py
│   ├── schemas/
│   │   ├── common.py
│   │   ├── rag.py
│   │   ├── recommendation.py
│   │   ├── task.py
│   │   └── user.py
│   ├── auth.py
│   ├── celery_app.py
│   ├── celery_metrics.py
│   ├── db.py
│   ├── db_models.py
│   ├── error_handlers.py
│   ├── main.py
│   ├── models.py
│   ├── services.py
│   └── utils/
│       ├── cv_model.py
│       ├── image_ops.py
│       ├── math_ops.py
│       └── torch_basic.py
├── alembic/
│   ├── versions/
│   └── env.py
├── tests/
│   ├── manual/
│   ├── models/
│   ├── nlp/
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_cv.py
│   ├── test_drift_detector.py
│   ├── test_maths.py
│   ├── test_ner_integration.py
│   ├── test_ner_service.py
│   └── test_tasks_crud.py
├── checkpoints/model.pth
├── data/
├── docker-compose.yml
├── Dockerfile
├── export_yolo_onnx.py
├── PORTFOLIO_TESTS_PLAN.md
├── prod_req.txt
├── prometheus.yml
├── pytest.ini
└── requirements.txt
```

---

## Быстрый запуск

### Docker

```bash
docker-compose up --build
```

API: <http://localhost:8000>

- Swagger: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- Ping: `GET /ping`

### Локально

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r prod_req.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Отдельно для фоновых задач:

```bash
celery -A app.celery_app worker -B --loglevel=info
```

---

## Полезные файлы

- `PORTFOLIO_TESTS_PLAN.md` — план необходимых тестов для портфолио.
- `requirements.txt` / `prod_req.txt` — зависимости.
- `docker-compose.yml` — локальная оркестрация API + Redis + Celery.
