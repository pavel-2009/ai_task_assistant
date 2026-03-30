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
- **Классификация**: инференс моделей классификатора (с чекпоинтами)
- **Детекция объектов**: YOLO v8/v10 (PyTorch & ONNX режимы)
- **Сегментация**: сегментация масок объектов
- **Embedding**: векторные представления изображений для поиска и рекомендаций

### Natural Language Processing (NLP) & RAG
- **Текстовые эмбеддинги**: SentenceTransformer (sentence-transformers/all-MiniLM-L6-v2)
- **Семантический поиск**: поиск по задачам на основе텍ста и изображений
- **NER (Named Entity Recognition)**: spacy (en_core_web_sm)
- **LLM интеграция**: локальная Ollama (llama3.2) с поддержкой streaming
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
- **Docker & docker-compose**: контейнеризированный запуск API + Redis + Celery + Ollama
- **Healthcheck**: автоматическая проверка готовности сервисов
- **Graceful shutdown**: корректное завершение и очистка ресурсов

---

## Стек

- **Backend:** FastAPI, Uvicorn
- **Auth:** JWT (PyJWT), OAuth2PasswordBearer, bcrypt
- **DB:** SQLite, SQLAlchemy Async, Alembic (инициализирован)
- **Async tasks:** Celery, Redis
- **LLM/RAG:** Ollama, FAISS, sentence-transformers
- **ML/CV:** PyTorch, torchvision, ultralytics, ONNX Runtime, OpenCV, Pillow
- **Infra:** Docker, docker-compose

---

## Структура проекта

```
app/
  ├── main.py                    # FastAPI приложение, lifespan, endpoints
  ├── auth.py                    # JWT, аутентификация, текущий пользователь
  ├── db.py                      # SQLAlchemy async engine, session
  ├── db_models.py               # ORM модели (User, Task)
  ├── models.py                  # Pydantic схемы
  ├── services.py                # Глобальный реестр инициализируемых сервисов
  ├── celery_app.py              # Конфигурация Celery, подключение моделей
  ├── error_handlers.py          # Глобальные обработчики исключений
  │
  ├── core/
  │  ├── config.py               # Переменные окружения (Settings)
  │  ├── dependencies.py         # Зависимости для роутеров (get_current_user и т.д.)
  │  ├── security.py             # Функции безопасности (хеширование паролей)
  │  └── rate_limit.py           # Rate limiting по пользователям
  │
  ├── routers/
  │  ├── auth.py                 # POST /auth/register, /auth/login
  │  ├── tasks.py                # GET/POST/PUT/DELETE /tasks
  │  ├── avatars.py              # Загрузка аватара, CV обработка
  │  ├── nlp.py                  # NLP эндпоинты (NER, embedding)
  │  ├── rag.py                  # RAG эндпоинты (поиск + генерация)
  │  ├── recsys.py               # Системы рекомендаций
  │  ├── monitoring.py           # Health-check, drift detection
  │  └── streaming.py            # WebSocket /ws/detect
  │
  ├── schemas/
  │  ├── task.py                 # Pydantic schema для Task
  │  ├── user.py                 # Pydantic schema для User
  │  ├── rag.py                  # Schemas для RAG запросов/ответов
  │  └── recommendation.py       # Schemas для рекомендаций
  │
  ├── ml/
  │  ├── common/                 # Общая конфигурация ML
  │  │  └── config.py            # Пути к чекпоинтам, гиперпараметры
  │  │
  │  ├── cv/                     # Computer Vision
  │  │  ├── tasks.py             # Celery tasks для CV
  │  │  ├── classification/      # Классификация изображений
  │  │  │  └── inference_service.py
  │  │  ├── detection/           # YOLO детекция (PyTorch + ONNX)
  │  │  │  ├── yolo_service.py   # PyTorch режим
  │  │  │  └── yolo_onnx_service.py # ONNX режим
  │  │  ├── embedding/           # Image embeddings
  │  │  │  └── image_embedding_service.py
  │  │  └── segmentation/        # Сегментация объектов
  │  │     └── segmentation_service.py
  │  │
  │  ├── nlp/                    # Natural Language Processing
  │  │  ├── tasks.py             # Celery tasks для NLP
  │  │  ├── embedding_service.py # SentenceTransformer эмбеддинги
  │  │  ├── ner_service.py       # Named Entity Recognition (spacy)
  │  │  ├── llm_service.py       # Ollama LLM интеграция
  │  │  ├── rag_service.py       # RAG логика (поиск + генерация)
  │  │  ├── semantic_search_service.py # Семантический поиск
  │  │  ├── vector_db.py         # FAISS vector database
  │  │
  │  ├── recsys/                 # Recommendation Systems
  │  │  ├── tasks.py             # Celery tasks для обучения
  │  │  ├── content_based.py     # Content-based рекомендации
  │  │  ├── collaborative_filtering.py # Collaborative filtering
  │  │  └── vector_db/           # Vector DB для RecSys
  │  │
  │  ├── monitoring/             # Мониторинг качества
  │  │  └── drift_detector.py    # Data drift detection
  │  │
  │  ├── avatars/                # Обработанные аватары (визуализ.)
  │  └── checkpoints/            # Сохраненные веса моделей
  │
  ├── utils/
  │  ├── cv_model.py             # Утилиты для CV моделей
  │  ├── image_ops.py            # Операции с изображениями
  │  ├── math_ops.py             # Математические операции
  │  └── torch_basic.py          # PyTorch утилиты
  │
  └── __init__.py

alembic/                         # Миграции БД
  ├── env.py                     # Конфигурация Alembic
  └── versions/                  # История миграций

tests/                           # Автотесты
  ├── conftest.py               # Fixtures и конфигурация pytest
  ├── test_auth.py              # Тесты авторизации
  ├── test_cv.py                # Тесты CV функций
  ├── test_ner_*.py             # Тесты NLP
  ├── test_maths.py             # Тесты математики
  ├── test_tasks_crud.py        # Тесты CRUD
  ├── test_drift_detector.py    # Тесты мониторинга
  ├── models/                   # Тесты для ML сервисов
  ├── nlp/                      # Тесты для NLP
  └── manual/                   # Ручные бенчмарки и тесты

data/                            # Данные для обучения и индексирования
  ├── data.yaml                 # YOLO dataset конфиг
  ├── vector_db.faiss           # Индекс FAISS для RAG
  ├── train/                    # Тренировочные данные
  └── valid/                    # Валидационные данные

alembic.ini                      # Конфиг Alembic
docker-compose.yml              # Docker Compose: API + Redis + Celery + Ollama
Dockerfile                      # Docker образ приложения
.env                            # Переменные окружения (не чекин в git)
prod_req.txt                    # Production зависимости
requirements.txt                # Development зависимости
pytest.ini                      # Конфиг pytest
README.md                       # Этот файл
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
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2
USE_ONNX=false
```

> **На Windows/Mac**, если Ollama установлена локально, может потребоваться:  
> `OLLAMA_BASE_URL=http://host.docker.internal:11434`

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

> Требуется **Python 3.11+** и установленная **Ollama** (для RAG функций).

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

# Ollama (если не запущена как сервис)
ollama serve
ollama pull llama3.2
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

> ⏳ **Важно:** Первый запуск займет 3–5 минут на загрузку ML моделей (~400MB+ за эмбеддинги, YOLO, Ollama).

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
- **ollama** — локальный LLM сервис на порту `11434`

Все контейнеры имеют **healthcheck** и ждут пока зависимости готовы.

### Просмотр логов

```bash
docker logs -f ai_task_assistant          # API
docker logs -f ai_task_assistant_celery   # Celery worker
docker logs -f ai_task_assistant_ollama   # Ollama
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
- **Ollama модель не загружена**: может занять 5–10 мин (~6GB для llama3.2)
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
- **Redis/Ollama недоступны**: проверьте healthcheck статус в `docker ps`
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
