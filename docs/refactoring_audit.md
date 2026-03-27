# Refactoring audit (API + project structure)

Ниже — список основных технических нюансов, которые мешают масштабировать проект и безопасно его развивать.

## 1) Смешение доменных слоёв в `app/models.py`

- В одном файле одновременно хранятся:
  - SQLAlchemy ORM-модели (`Task`, `User`, `Text`),
  - Pydantic API-схемы (`TaskCreate`, `TaskGet`, `UserGet`, `RecommendationGet`).
- Это нарушает разделение ответственности и усложняет миграцию на SQLAlchemy 2.0 style / отдельный API contract.

### Рекомендация

- Разделить на пакеты:
  - `app/db/models/*.py` — только ORM,
  - `app/schemas/*.py` — только Pydantic.
- В роутерах импортировать только схемы, в репозиториях/сервисах — ORM.

## 2) Pydantic-модели прямо внутри роутеров

- В `app/routers/rag.py` ранее `AskRequest` объявлялся прямо в роутере.
- При росте API такие объявления затрудняют переиспользование схем, генерацию SDK и типизацию.

### Что уже сделано

- `AskRequest` вынесен в `app/schemas/rag.py`.
- Роутер теперь импортирует схему из `app.schemas`.

## 3) Роутеры содержат бизнес-логику и инфраструктурные детали

Примеры:
- `tasks.py`: orchestration Celery + прямые SQL операции + подготовка response.
- `avatars.py`: файловая система, валидация изображения, Celery, SQL в одном модуле.

### Рекомендация

- Ввести слои:
  - `services/` — бизнес-операции,
  - `repositories/` — SQL и трансакционные операции,
  - `routers/` — только HTTP-адаптер (валидация, коды, вызов service).

## 4) Несогласованные URL и префиксы

- В `tasks` роутере есть endpoint `"/tasks/{task_id}/tags_status"` при `prefix="/tasks"`.
- Фактический путь становится `"/tasks/tasks/{task_id}/tags_status"`.

### Рекомендация

- Привести к `"/{task_id}/tags-status"` или `"/{task_id}/tags_status"`.
- Принять единый стиль именования URL (`kebab-case` предпочтительно).

## 5) Антипаттерн обработки ошибок

- В `nlp.py` и `rag.py` встречается `except Exception as e` с возвратом 500 + `str(e)` клиенту.
- Это раскрывает внутренние детали и затрудняет observability.

### Рекомендация

- Ввести централизованный error-handling:
  - кастомные исключения доменного слоя,
  - exception handlers FastAPI,
  - маскирование внутренних ошибок для клиента.

## 6) Неконсистентная работа с nullability и валидацией схем

- В `TaskBase` поля `avatar_file`/`tags` типизированы как `str`, но default `None`.
- Семантически это `str | None`.

### Рекомендация

- Явно указать Optional типы: `avatar_file: str | None`, `tags: str | None`.
- У `TaskUpdate` сделать поля частично optional (patch semantics), а не наследовать полностью required `TaskBase`.

## 7) Роутеры перегружены зависимостями через `request.app.state`

- Сервисы получаются через helper-функции в роутерах, без единой DI точки.

### Рекомендация

- Вынести провайдеры зависимостей в `app/dependencies.py`.
- В роутерах использовать `Depends(get_rag_service)` и т.п.

## 8) Неровная модульная организация

- `app/services.py` и ML-папки решают разные уровни задач.
- Есть потенциальное смешение API/domain/ML orchestration.

### Рекомендация

- Выделить bounded-context пакеты:
  - `app/domains/tasks/*`,
  - `app/domains/auth/*`,
  - `app/domains/recsys/*`.
- В каждом контексте хранить `schemas`, `service`, `repository`, `router` (или adapter).

## Предлагаемый поэтапный план

1. **Схемы и ORM развести по разным пакетам** (минимальный риск).
2. **Вынести бизнес-логику из роутеров в сервисы** без изменения API контрактов.
3. **Унифицировать ошибки и DI**.
4. **Нормализовать URL и naming conventions**.
5. **Далее декомпозировать по доменам**.
