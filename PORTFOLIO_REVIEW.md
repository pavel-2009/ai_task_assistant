# Portfolio Fixes Checklist (Tech Demo Project)

Этот документ — чеклист улучшений для приведения проекта к аккуратному и инженерно-зрелому виду без изменения его сути как тренировочного проекта.

---

# 🔴 1. Архитектурная ясность

## 1.1 Добавить ARCHITECTURE.md

Создать файл `ARCHITECTURE.md` со структурным описанием:

# Architecture Overview

## Layers

- routers → HTTP слой (FastAPI endpoints)
- services → бизнес-логика
- ml → ML inference / training
- utils → чистые утилиты (без бизнес-логики)

## Modules purpose

- auth → JWT авторизация и безопасность
- tasks → CRUD и управление задачами
- nlp → embeddings, semantic search, RAG
- cv → обработка изображений (classification/detection)
- recsys → рекомендательные алгоритмы
- celery → асинхронные задачи
- monitoring → метрики и healthchecks

---

## 1.2 Зафиксировать правила слоёв

Добавить:

## Layer Rules

- routers НЕ содержат бизнес-логики
- services НЕ знают про HTTP
- ml изолирован от API
- utils не работают с БД и внешними сервисами

---

# 🔴 2. Декомпозиция “god files”

## 2.1 Разбить services.py

Было:

app/services.py

Сделать:

app/services/
  user_service.py
  task_service.py
  auth_service.py

---

## 2.2 Разбить models.py

Было:

app/models.py

Сделать:

app/models/
  user.py
  task.py

---

# 🟠 3. Стандартизация ML слоя

## 3.1 Ввести базовый интерфейс

class BaseMLService:
    def predict(self, data):
        raise NotImplementedError

Использовать в:
- classification
- detection
- embedding

---

## 3.2 Унифицировать naming

Принять правило:

- *_service.py → inference логика
- *_tasks.py → celery задачи
- *_utils.py → вспомогательные функции

---

## 3.3 Убрать дублирование

Объединить:

- yolo_service
- yolo_onnx_service

В:

YoloService(provider="torch" | "onnx")

---

# 🟡 4. Очистка репозитория

## 4.1 Удалить из git

- *.pt
- *.onnx
- data/
- checkpoints/

---

## 4.2 Обновить .gitignore

*.pt
*.onnx
data/
checkpoints/

---

## 4.3 Добавить заглушки

data/README.md

# Dataset is not included in the repository

---

# 🟢 5. README улучшения

## 5.1 Добавить блок “What this project demonstrates”

## What this project demonstrates

- FastAPI архитектура (routers/services separation)
- JWT authentication и security
- Асинхронные задачи (Celery + Redis)
- ML интеграция (CV + NLP)
- Vector search (FAISS)
- Monitoring (Prometheus)

---

## 5.2 Добавить блок тестов

## Tests

Run tests:

pytest

With coverage:

pytest --cov=app

---

## 5.3 Добавить описание модулей

## Modules

- auth — авторизация
- tasks — CRUD задач
- nlp — обработка текста
- cv — обработка изображений
- recsys — рекомендации
- monitoring — метрики

---

# 🔵 6. Docker и окружение

## 6.1 Добавить PostgreSQL в docker-compose

postgres:
  image: postgres:15
  environment:
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: password
    POSTGRES_DB: postgres

---

## 6.2 Привести .env.example к одному стилю

Было:

POSTGRES_USER: postgres

Должно быть:

POSTGRES_USER=postgres

---

# 🟣 7. Тесты как часть демонстрации

## 7.1 Добавить в README

- описание типов тестов (unit / integration)
- как запускать

---

## 7.2 (Опционально) добавить coverage badge

---

# ⚫ 8. Code style и консистентность

## 8.1 Добавить инструменты

- ruff
- black
- mypy

---

## 8.2 Привести naming к одному стилю

Проблемные файлы:

- math_ops.py
- torch_basic.py

Рекомендуется:

numeric_utils.py
torch_utils.py

---

# ⚪ 9. Monitoring как фича

Добавить в README:

## Monitoring

- /metrics endpoint (Prometheus)
- сбор метрик API и Celery

---

# 🔥 10. Итоговый чеклист

## MUST

- [ ] ARCHITECTURE.md
- [ ] разбит services.py
- [ ] разбит models.py
- [ ] очищен репозиторий от моделей и данных
- [ ] добавлен блок “What this project demonstrates”

---

## NICE TO HAVE

- [ ] PostgreSQL в docker-compose
- [ ] единый стиль .env
- [ ] базовый ML интерфейс
- [ ] улучшенный README
- [ ] code style инструменты

---

# 🧠 Цель изменений

Сейчас проект демонстрирует:
“много технологий”

После исправлений будет демонстрировать:
“контроль сложности, архитектурное мышление и инженерную аккуратность”