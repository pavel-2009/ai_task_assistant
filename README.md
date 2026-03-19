# Виртуальный AI‑планировщик задач

FastAPI‑сервис для управления задачами с авторизацией, загрузкой аватаров и CV/ML‑обработкой изображений (классификация, детекция, сегментация), включая фоновые задачи через Celery и Redis.

---

## Что уже реализовано

- REST API на **FastAPI**:
  - регистрация и вход пользователя (JWT)
  - CRUD задач
  - загрузка аватара задачи
  - запуск ML‑задач в фоне
- Асинхронная БД на **SQLAlchemy + aiosqlite**.
- Фоновые задачи на **Celery + Redis**.
- ML/CV‑часть:
  - инференс классификатора
  - YOLO‑детекция (PyTorch / ONNX)
  - сегментация изображения
- WebSocket‑эндпоинт для realtime‑детекции.
- Dockerized запуск API + Redis + Celery.

---

## Стек

- **Backend:** FastAPI, Uvicorn
- **Auth:** JWT (PyJWT), OAuth2PasswordBearer, bcrypt
- **DB:** SQLite, SQLAlchemy Async, Alembic (инициализирован)
- **Async tasks:** Celery, Redis
- **ML/CV:** PyTorch, torchvision, ultralytics, ONNX Runtime, OpenCV, Pillow
- **Infra:** Docker, docker-compose

---

## Структура проекта

```text
app/
  main.py                # Точка входа FastAPI
  auth.py                # JWT/пароли/текущий пользователь
  db.py                  # Async engine/session
  models.py              # SQLAlchemy + Pydantic схемы
  celery_app.py          # Конфиг Celery + preload моделей
  routers/
    auth.py              # /auth
    tasks.py             # /tasks
    avatars.py           # эндпоинты аватаров + CV флоу
    streaming.py         # /ws/detect
  ml/
    common/              # общая ML-конфигурация
    cv/                  # CV-подпакет
      classification/    # классификация изображений
      detection/         # детекция объектов и YOLO-веса
      segmentation/      # сегментация изображений
    nlp/                 # NLP-сервисы и векторный поиск

alembic/                 # Конфигурация миграций
tests/                   # Автотесты
docker-compose.yml       # API + Redis + Celery
Dockerfile               # Образ приложения
```

---

## Быстрый старт (Docker)

### 1) Подготовить `.env`

Минимум:

```env
SECRET_KEY=change_me
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
USE_ONNX=False
```

### 2) Собрать и поднять сервисы

```bash
docker compose up --build
```

После запуска API доступен на `http://localhost:8000`.

- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health-check: `GET /ping`

---

## Локальный запуск (без Docker)

> Требуется Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r prod_req.txt
uvicorn app.main:app --reload
```

Для фоновых задач отдельно поднимите Redis и Celery worker:

```bash
celery -A app.ml.cv.tasks worker --loglevel=info
```

---

## Базовый API‑флоу

1. **Регистрация**: `POST /auth/register`
2. **Логин**: `POST /auth/login` → получить `access_token`
3. **Создать задачу**: `POST /tasks/` (Bearer token)
4. **Загрузить аватар**: endpoint из `avatars`‑роутера
5. **Запустить CV‑задачу** (predict/detect/segment) и опрашивать status endpoint

> Актуальные пути и контракты — в Swagger (`/docs`).

---

## Тесты

Запуск:

```bash
pytest -q
```

Если тесты не стартуют на чистом окружении, обычно причина — отсутствующие зависимости (например, `fastapi`, `numpy`, `ultralytics`, `websockets`) или неподнятые инфраструктурные сервисы.

---

## Известные зоны доработки

- Выравнивание и стабилизация API‑контрактов в роутерах.
- Усиление lifecycle/logging/ошибок на старте приложения.
- Расширение тестов (контрактные, интеграционные, e2e).
- Приведение миграций к полноценному рабочему циклу.
- Улучшение документации по deployment/monitoring.

---

## Roadmap (предложение)

- [ ] Зафиксировать и версионировать API‑контракты.
- [ ] Добавить CI (линтер + тесты + smoke).
- [ ] Покрыть критические сценарии интеграционными тестами.
- [ ] Добавить наблюдаемость (структурные логи, метрики, трассировка).
- [ ] Подготовить production‑профиль (secrets, health/readiness, retries/timeouts).

---

## Лицензия

Пока не указана.
