# Виртуальный AI‑планировщик задач

FastAPI‑сервис для управления задачами с авторизацией, загрузкой аватаров, NLP/RAG и CV/ML‑обработкой изображений (классификация, детекция, сегментация), включая фоновые задачи через Celery и Redis.

---

## Что уже реализовано

### REST API на FastAPI
- **Авторизация & Пользователи**: JWT (PyJWT), регистрация, логин, управление сессиями
- **CRUD задач**: создание, чтение, обновление, удаление задач
- **Управление аватарами**: загрузка и обработка изображений задач
- **Rate limiting**: ограничение частоты запросов по пользователю

### База данных
- **SQLAlchemy + aiosqlite**: асинхронная работа с SQLite
- **Alembic**: миграции БД (инициализирован)
- **db_models.py**: модели пользователей и задач

### Фоновые задачи & Кеширование
- **Celery + Redis**: асинхронное выполнение ML-задач
- **Beat scheduler**: периодические задачи (переиндексация, переобучение)

### Computer Vision (CV)
- **Классификация**: инференс ResNet-классификатора на 1000 классах ImageNet (с чекпоинтом)
- **Детекция объектов**: YOLO v8/v10 (PyTorch & ONNX режимы)
- **Сегментация**: сегментация масок объектов
- **Embedding**: векторные представления изображений для поиска и рекомендаций

### Natural Language Processing (NLP) & RAG
- **Текстовые эмбеддинги**: SentenceTransformer (sentence-transformers/all-MiniLM-L6-v2)
- **Семантический поиск**: поиск по задачам на основе текста и изображений
- **NER (Named Entity Recognition)**: spacy (en_core_web_sm)
- **LLM интеграция**: облачная OpenRouter-совместимая LLM (free-tier модели) с поддержкой streaming
- **RAG (Retrieval-Augmented Generation)**: поиск релевантных задач + генерация ответов LLM
- **Векторная БД**: FAISS для быстрого поиска эмбеддингов (с Redis кешированием)

### Системы рекомендаций (RecSys)
- **Content-Based**: рекомендации на основе схожести эмбеддингов (текст + изображение)
- **Collaborative Filtering**: рекомендации на основе взаимодействий пользователей

### Мониторинг & Quality Assurance
- **Data Drift Detection**: детектор дрейфа данных для аватаров
- **Health-check endpoints**: `/ping` для проверки статуса приложения и сервисов

### WebSocket & Real-time
- WebSocket-эндпоинт для real-time детекции объектов в потоке видео

### Инфраструктура
- **Docker & docker-compose**: контейнеризированный запуск API + Redis + Celery
- **Healthcheck**: автоматическая проверка готовности сервисов
- **Graceful shutdown**: корректное завершение и очистка ресурсов

---

## Стек

- **Backend:** FastAPI, Uvicorn
- **Auth:** JWT (PyJWT), OAuth2PasswordBearer, bcrypt
- **DB:** SQLite, SQLAlchemy Async, Alembic (инициализирован)
- **Async tasks:** Celery, Redis
- **LLM/RAG:** OpenRouter (или другой OpenAI-compatible API), FAISS, sentence-transformers
- **ML/CV:** PyTorch, torchvision, ultralytics, ONNX Runtime, OpenCV, Pillow
- **Infra:** Docker, docker-compose

---

## Структура проекта

```text
app/
  ├── main.py                     # FastAPI приложение, lifespan, маршруты
  ├── auth.py                     # JWT и вспомогательная логика аутентификации
  ├── db.py                       # SQLAlchemy async engine/session
  ├── db_models.py                # ORM модели (User, Task, Interaction)
  ├── models.py                   # Базовые Pydantic-модели API
  ├── services.py                 # Реестр и инициализация ML/NLP/CV сервисов
  ├── celery_app.py               # Конфигурация Celery
  ├── celery_metrics.py           # Метрики Celery задач
  ├── error_handlers.py           # Глобальные обработчики исключений
  │
  ├── core/
  │  ├── config.py                # Settings и env-конфигурация приложения
  │  ├── dependencies.py          # FastAPI dependencies
  │  ├── metrics.py               # Prometheus/служебные метрики API
  │  ├── rate_limit.py            # Ограничение частоты запросов
  │  └── security.py              # Хеширование паролей и security helpers
  │
  ├── routers/
  │  ├── auth.py                  # /auth/*
  │  ├── tasks.py                 # /tasks CRUD
  │  ├── avatars.py               # /avatars (загрузка/обработка)
  │  ├── nlp.py                   # /nlp endpoints
  │  ├── rag.py                   # /rag endpoints
  │  ├── recsys.py                # /recsys endpoints
  │  ├── monitoring.py            # /ping, /metrics, drift checks
  │  └── streaming.py             # WebSocket /ws/detect
  │
  ├── schemas/
  │  ├── common.py                # Общие схемы ответов
  │  ├── task.py                  # Схемы задач
  │  ├── user.py                  # Схемы пользователей
  │  ├── rag.py                   # Схемы RAG
  │  └── recommendation.py        # Схемы рекомендаций
  │
  ├── ml/
  │  ├── common/config.py         # ML-конфиг (включая num_classes=1000)
  │  ├── metrics.py               # Метрики ML-сервисов
  │  ├── cv/
  │  │  ├── tasks.py
  │  │  ├── classification/       # datasets, train_loop, inference_service, models_nn
  │  │  ├── detection/            # YOLO (PyTorch, ONNX, export)
  │  │  ├── embedding/            # image_embedding_service
  │  │  └── segmentation/         # segmentation_service
  │  ├── nlp/                     # embedding, NER, LLM, RAG, vector_db, tasks
  │  ├── recsys/                  # content_based, collaborative_filtering, celery tasks
  │  └── monitoring/              # drift_detector
  │
  └── utils/
     ├── cv_model.py
     ├── image_ops.py
     ├── math_ops.py
     └── torch_basic.py

alembic/
  ├── env.py
  └── versions/

tests/
  ├── conftest.py
  ├── test_auth.py
  ├── test_tasks_crud.py
  ├── test_cv.py
  ├── test_ner_service.py
  ├── test_ner_integration.py
  ├── test_drift_detector.py
  ├── test_maths.py
  ├── nlp/                        # unit/integration тесты RAG/vector DB
  ├── models/                     # тесты ML/CV сервисов
  └── manual/                     # ручные smoke/benchmark скрипты

data/                             # датасеты и артефакты для CV
checkpoints/model.pth             # чекпоинт классификатора
Dockerfile                        # Docker-образ API
docker-compose.yml                # API + Redis + Celery
prometheus.yml                    # Конфиг Prometheus
requirements.txt                  # dev-зависимости
prod_req.txt                      # prod-зависимости
PORTFOLIO_REVIEW.md               # Актуальный чеклист для портфолио
README.md                         # Документация проекта
```

---

## Быстрый старт (Docker)

### 1) Проверить `.env`

Убедитесь, что файл `.env` содержит необходимые переменные:

```env
SECRET_KEY=your-secret-key-here-change-in-production-12345
DATABASE_URL=sqlite+aiosqlite:///./test.db
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.3-8b-instruct:free
LLM_API_KEY=your-openrouter-api-key
USE_ONNX=false
```

> Для облачной модели укажите `LLM_API_KEY` и (опционально) `LLM_MODEL` в `.env`.

### 2) Собрать и запустить сервисы

```bash
docker-compose up
```

⏳ **Первый запуск займет 3-5 минут** для загрузки ML моделей (~400MB+ за эмбеддинги и веса моделей).

### 3) Проверить доступность

API доступен на **http://localhost:8000**:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health-check**: `curl http://localhost:8000/ping`

Ожидаемый ответ при полной готовности:

```json
{
  "status": "ok",
  "services_initialized": true,
  "models_ready": true,
  "models": {
    "embedding": {"ready": true, ...},
    "vector_db": {"ready": true, ...},
    "ner": {"ready": true, ...},
    "llm": {"ready": true, ...}
  }
}
```

> **Если `services_initialized=false`**, приложение все еще инициализируется. Проверьте логи:  
> `docker logs -f ai_task_assistant`

---

## Локальный запуск (без Docker)

> Требуется **Python 3.11+** и ключ к облачному LLM API (например, OpenRouter) для RAG функций.

### 1) Подготовить окружение

```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2) Установить зависимости

```bash
pip install -r prod_req.txt
```

### 3) Запустить зависимости

В отдельных терминалах:

```bash
# Redis
redis-server

# Cloud LLM запускается вне проекта (проверьте LLM_API_KEY в .env)
```

### 4) Запустить приложение

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5) В отдельном терминале - Celery worker

```bash
celery -A app.celery_app worker -B --loglevel=info
```

API доступен: **http://localhost:8000**  
Swagger: **http://localhost:8000/docs**

---

## Запуск в Docker

> ⏳ **Важно:** Первый запуск займет 3–5 минут на загрузку ML моделей (~400MB+ за эмбеддинги и YOLO).

### Быстрый старт

```bash
docker-compose up
```

Приложение будет доступно по адресу **http://localhost:8000** как только контейнеры полностью инициализируются.

### Проверка статуса

```bash
# В отдельном терминале проверяйте readiness
curl http://localhost:8000/ping
```

**Ожидаемый ответ:**

```json
{"status": "ok", "services_initialized": true, "models_ready": true}
```

### Что включено в Docker Compose

- **api** — FastAPI приложение на порту `8000`
- **celery** — фоновые задачи и scheduler
- **redis** — кеш и message broker для Celery
- **внешний облачный LLM API** — используется через переменные окружения `LLM_*`

Все контейнеры имеют **healthcheck** и ждут пока зависимости готовы.

### Просмотр логов

```bash
docker logs -f ai_task_assistant          # API
docker logs -f ai_task_assistant_celery   # Celery worker
docker logs -f ai_task_assistant_redis    # Redis
```

### Остановка

```bash
docker-compose down
```

Для полной переинициализации с чистого листа:

```bash
docker-compose down -v
```

### Часто встречаемые проблемы

Если приложение стартует, но не откуется в браузере → [см. DOCKER_TROUBLESHOOTING.md](DOCKER_TROUBLESHOOTING.md)

- **Redis недоступен**: дождитесь статуса ✓ healthy в `docker ps`
- **LLM API недоступен**: проверьте `LLM_API_KEY`, `LLM_BASE_URL` и лимиты аккаунта провайдера
- **"Services still initializing"**: посмотрите `docker logs ai_task_assistant | grep -i error`

---

## Базовый API‑флоу

## Базовый API‑флоу

### Пример workflow

1. **Регистрация**: `POST /auth/register`
   ```json
   {"username": "user1", "password": "secure_pass"}
   ```

2. **Логин**: `POST /auth/login`
   ```json
   {"username": "user1", "password": "secure_pass"}
   ```
   → Получить `access_token`

3. **Создать задачу**: `POST /tasks/` (с Bearer token)
   ```json
   {"title": "Найти кота", "description": "На фото хозяина"}
   ```

4. **Загрузить аватар**: `POST /avatars/upload/{task_id}`
   → Запустится классификация/детекция

5. **Семантический поиск**: `POST /rag/search`
   ```json
   {"query": "Найди похожие задачи про животных"}
   ```

6. **RAG-вопрос**: `POST /rag/ask`
   ```json
   {"query": "Какие задачи связаны с поиском потерянных предметов?"}
   ```
   → LLM вернет ответ с контекстом из других задач

7. **Рекомендации**: `GET /recsys/recommend/{task_id}`
   → Система порекомендует похожие задачи

8. **WebSocket детекция**: `WS /ws/detect`
   → Real-time детекция объектов в видеопотоке

> 📖 Полная документация API: **http://localhost:8000/docs** (Swagger UI)

---

## Тесты

Запуск:

```bash
pytest -q
```

Если тесты не стартуют на чистом окружении, обычно причина — отсутствующие зависимости (например, `fastapi`, `numpy`, `ultralytics`, `websockets`) или неподнятые инфраструктурные сервисы.

---

## Известные вопросы и возможности улучшения

- **Docker контейнер не стартует**: см. [DOCKER_TROUBLESHOOTING.md](DOCKER_TROUBLESHOOTING.md)
- **ML модели медленно загружаются**: первый запуск скачивает ~400MB+ моделей. Это нормально.
- **Redis/LLM API недоступны**: проверьте healthcheck Redis и корректность `LLM_*` переменных
- **CUDA/GPU не используется**: USE_ONNX должна быть `false`, проверьте наличие CUDA в системе
- **Миграции БД**: Alembic подготовлен, но требует проверки в боевой среде
- **Тесты**: есть базовые тесты, рекомендуется расширить интеграционные сценарии

## Roadmap (предложение)

- [ ] Настроить CI/CD (GitHub Actions): линтер, тесты, docker build
- [ ] Расширить тестовое покрытие (интеграционные, e2e тесты для CV/NLP)
- [ ] Добавить структурированное логирование (Structlog + JSON)
- [ ] Добавить метрики (Prometheus: latency, model inference time, cache hit rate)
- [ ] Оптимизировать загрузку моделей (lazy loading, шаринг весов между моделями)
- [ ] Добавить кеширование результатов классификации/детекции по хешу изображения
- [ ] Настроить graceful shutdown для фоновых задач
- [ ] Подготовить production-профиль (secrets management, health/readiness probes)
- [ ] Документировать API контракты (OpenAPI schema versioning)
- [ ] Добавить поддержку batch processing для CV моделей

---

## Лицензия

Пока не указана.
