# Ревью проекта для портфолио (актуальные замечания)

Ниже — только те пункты, которые остаются актуальными после внесенных исправлений.

## 1) `app/ml/recsys/tasks.py` + `app/ml/recsys/collaborative_filtering.py`

- **Согласовать формат сериализуемой модели collaborative filtering.**  
  В задаче обучения в Redis кладется `(user_item_matrix, user_to_idx, task_to_idx, unique_users, unique_tasks, popular_tasks)`, а метод загрузки ожидает `(user_factors, item_factors, user_to_idx, task_to_idx, idx_to_task, popular_tasks)`. В текущем виде рекомендации после обучения работают на несовместимом формате и логика `recommend` некорректна.  
  **Где:** `app/ml/recsys/tasks.py:138`, `app/ml/recsys/collaborative_filtering.py:45-51,64-83`.

## 2) `app/ml/cv/classification/inference_service.py` + `app/core/config.py` + `app/ml/common/config.py`

- **Согласовать классы классификации с моделью чекпоинта.**  
  Классификатор строится на `num_classes=3`, но в `Settings` подставляется словарь `INFERENCE_IDX_TO_CLASS` из 1000 классов ImageNet. В текущей конфигурации предсказания/лейблы логически несогласованы и выглядят как “случайный” mapping.  
  **Где:** `app/ml/common/config.py:19`, `app/core/config.py:35`, `app/ml/cv/classification/inference_service.py:22-25,64-66`.

## 3) `tests/` + окружение

- **Починить воспроизводимость тестового окружения.**  
  На чистом окружении тесты не стартуют даже на этапе collection (`ModuleNotFoundError: pydantic`, `fastapi`). Для портфолио проект должен подниматься одной командой в venv/docker и запускать хотя бы основной smoke-test набор.  
  **Где проверено командой:** `pytest -q`.
