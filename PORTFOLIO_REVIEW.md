# Ревью проекта для портфолио (что нужно исправить)

Ниже — список исправлений, которые нужно внести, чтобы проект выглядел как цельный и надежный production-like pet-проект, а не набор разрозненных демо-фич.

## 1) `app/main.py`

- **Починить запуск приложения: неверный аргумент при инициализации сервисов.**  
  Сейчас вызывается `ensure_services_initialized(..., idx_to_class=...)`, но функция принимает `inference_idx_to_class`. Из-за этого приложение падает на старте с `TypeError`.  
  **Где:** `app/main.py:50-53`, `app/services.py:196-215`.

- **Убрать shutdown Celery из API-процесса.**  
  В `lifespan` вызывается `celery_app.control.shutdown(timeout=30)`. Это останавливает воркеры глобально и может ломать соседние инстансы/окружения, если воркер общий. API не должен завершать lifecycle чужих процессов.  
  **Где:** `app/main.py:106-110`.

## 2) `app/ml/nlp/tasks.py`

- **Исправить асинхронные вызовы RecSys Vector DB (`await` обязателен).**  
  `rs_vector_db.update(...)` и `rs_vector_db.add_vector(...)` объявлены async, но вызываются без `await` — код фактически не выполняет обновление индекса.  
  **Где:** `app/ml/nlp/tasks.py:151-155`, `app/ml/recsys/vector_db/recsys_vector_db.py:27-45,154-167`.

- **Исправить размерность объединенного эмбеддинга в рекомендациях.**  
  Для задачи без аватара дописывается `torch.zeros(512)`, но в сервисе зашита размерность `896` (`384` текст + `512` изображение). При наличии аватара конкатенируется текст + *сырое* изображение из другого пайплайна с другой размерностью (см. ниже), что ломает размерность и поиск.  
  **Где:** `app/ml/nlp/tasks.py:136,141-143`, `app/services.py:137-145`, `app/ml/cv/embedding/image_embedding_service.py:25,48`.

- **Убрать дублирующую запись текста в таблицу `texts`.**  
  В `_process_task_tags_and_embedding_async` сначала вызывается `semantic_search_service.index(...)`, который уже делает `insert(Text)`, а затем идет еще один `insert(Text)` с тем же `text_id`. Это приводит к конфликтам уникальности/ошибкам на повторной индексации.  
  **Где:** `app/ml/nlp/tasks.py:53-64`, `app/ml/nlp/vector_db.py:66-68`.

- **Привести типы идентификаторов к одному виду.**  
  Векторная БД NLP хранит `text_id` как `str`, а в задаче в `item_id` передается `int` (`task_id`), плюс в другом месте идет проверка `if task.id not in semantic_search_service.vector_db.ids`, где `task.id` — int, `ids` — строки. Из-за этого реиндексация будет повторно добавлять те же записи.  
  **Где:** `app/ml/nlp/tasks.py:96-101`, `app/ml/nlp/vector_db.py:33,45,57`.

## 3) `app/ml/recsys/content_based.py`

- **Починить логику поиска похожих задач (сейчас она структурно сломана).**  
  `search` возвращает список `(task_id, score)`, но далее этот список напрямую передается в `Task.id.in_(tasks)`, что невалидно для SQL-фильтра по ID. После `scalars().all()` вы получаете список `Task`, но код итерируется как `for task, score in tasks`, что упадет.  
  **Где:** `app/ml/recsys/content_based.py:110,117-121`.

- **Исправить фильтрацию по `author_id` и возврат результатов.**  
  Добавление в `similar_tasks` происходит только внутри блока `if author_id is not None`; при `author_id=None` метод возвращает пустой список даже при найденных похожих задачах.  
  **Где:** `app/ml/recsys/content_based.py:121-133`.

- **Починить работу без Redis.**  
  В `_get_task_embedding` используется `await self.recsys_vector_db.redis_client.get(...)` без проверки на `None`, что ломает поток при отсутствии Redis (а в проекте явно предусмотрен degraded mode без Redis).  
  **Где:** `app/ml/recsys/content_based.py:62,74`.

## 4) `app/ml/recsys/tasks.py` + `app/ml/recsys/collaborative_filtering.py`

- **Согласовать формат сериализуемой модели collaborative filtering.**  
  В задаче обучения в Redis кладется `(user_item_matrix, user_to_idx, task_to_idx, unique_users, unique_tasks, popular_tasks)`, а метод загрузки ожидает `(user_factors, item_factors, user_to_idx, task_to_idx, idx_to_task, popular_tasks)`. В текущем виде рекомендации после обучения работают на несовместимом формате и логика `recommend` некорректна.  
  **Где:** `app/ml/recsys/tasks.py:138`, `app/ml/recsys/collaborative_filtering.py:45-51,64-83`.

## 5) `app/ml/nlp/semantic_search_service.py`

- **Удалить или реализовать несуществующие sync-методы.**  
  `index_sync` вызывает `vector_db.add_sync`, `save_to_redis_sync`, `clear_cache_sync`, которых в классе `VectorDB` нет. Это мертвый/ломающийся API и плохой сигнал для ревьюера.  
  **Где:** `app/ml/nlp/semantic_search_service.py:60-73`, `app/ml/nlp/vector_db.py` (таких методов нет).

## 6) `app/routers/nlp.py` + `app/ml/nlp/ner_service.py` + `app/schemas/common.py`

- **Согласовать контракт ответа `/nlp/tag-task`.**  
  Схема `NLPTagTaskResponse` ожидает `tags: list[str]`, но `ner_service.tag_task()` возвращает dict вида `{technologies, confidence}`. Сейчас endpoint возвращает объект не по схеме — это либо 500, либо некорректная сериализация.  
  **Где:** `app/routers/nlp.py:151-169`, `app/ml/nlp/ner_service.py:92-101`, `app/schemas/common.py:155-158`.

## 7) `app/routers/rag.py` + `app/ml/nlp/rag_service.py` + `app/ml/nlp/vector_db.py` + `app/schemas/common.py`

- **Привести к одному формату данные поиска для RAG.**  
  `RAGService` форматирует задачи как `title/description/tags/task_id`, но `semantic_search`/`vector_db` возвращает `text_id/text/similarity`. В результате источники и форматирование ответа теряют ключевые поля или дают пустые значения.  
  **Где:** `app/ml/nlp/rag_service.py:35-44,49-57`, `app/ml/nlp/vector_db.py:159-161`, `app/schemas/common.py:60-67`.

- **Пересобрать схему search-результатов NLP.**  
  `SearchResultItem` описан как `task_id/title/description/score`, но фактический поиск возвращает другую структуру (`text_id`, `text`, `similarity`). Схемы и реальный payload должны совпадать.  
  **Где:** `app/schemas/common.py:60-67`, `app/ml/nlp/semantic_search_service.py:87-96`, `app/ml/nlp/vector_db.py:159-161`.

## 8) `app/ml/cv/classification/inference_service.py` + `app/core/config.py` + `app/ml/common/config.py`

- **Согласовать классы классификации с моделью чекпоинта.**  
  Классификатор строится на `num_classes=3`, но в `Settings` подставляется словарь `INFERENCE_IDX_TO_CLASS` из 1000 классов ImageNet. В текущей конфигурации предсказания/лейблы логически несогласованы и выглядят как “случайный” mapping.  
  **Где:** `app/ml/common/config.py:19`, `app/core/config.py:35`, `app/ml/cv/classification/inference_service.py:22-25,64-66`.

## 9) `app/ml/cv/embedding/image_embedding_service.py` + `app/services.py`

- **Привести image embedding к стабильной фиксированной размерности и единому pipeline.**  
  Сейчас вырезается backbone до `[:-2]`, то есть тензор признаков имеет пространственную размерность (не вектор фиксированного размера), но RecSys ожидает fixed dim (`896`). Нужно сделать единый контракт (например, GAP -> 2048, потом projection до нужной размерности) и использовать его везде одинаково.  
  **Где:** `app/ml/cv/embedding/image_embedding_service.py:25,48`, `app/services.py:137-145`.

## 10) `tests/` + окружение

- **Починить воспроизводимость тестового окружения.**  
  На чистом окружении тесты не стартуют даже на этапе collection (`ModuleNotFoundError: pydantic`, `fastapi`). Для портфолио проект должен подниматься одной командой в venv/docker и запускать хотя бы основной smoke-test набор.  
  **Где проверено командой:** `pytest -q`.

---

## Что в итоге должно получиться после исправлений

После исправления пунктов выше у проекта будет:
- рабочий запуск без падения на старте;
- непротиворечивая логика NLP/CV/RecSys по контрактам данных;
- воспроизводимый pipeline индексации и рекомендаций;
- корректные схемы API и ответы, совпадающие с реальными payload;
- адекватная демонстрация прикладной логики для портфолио без «сломанных демо-фич».
