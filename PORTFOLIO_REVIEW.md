# Ревью проекта для портфолио (актуальные замечания)

Ниже — только те пункты, которые остаются актуальными после последних изменений.

## 1) `app/ml/recsys/tasks.py` + `app/ml/recsys/collaborative_filtering.py`

- **Согласовать формат сериализуемой модели collaborative filtering.**  
  В задаче обучения в Redis кладется `(matrix, user_to_idx, idx_to_task, unique_users, unique_tasks, popular_tasks)`, и этот формат частично используется в `load()`, но сама логика `recommend()` вычисляет `scores = matrix.dot(user_vector)`, где размеры/тип результата для user-item матрицы не соответствуют ожидаемому вектору скорингов по задачам. В текущем виде рекомендации после обучения могут быть некорректными.
  **Где:** `app/ml/recsys/tasks.py`, `app/ml/recsys/collaborative_filtering.py`.

## 2) `app/ml/cv/classification/*` + `app/core/config.py`

- **Пункт про количество классов закрыт: сейчас используется 1000 классов ImageNet.**  
  `MLConfig.num_classes` берется из `ResNet18_Weights.IMAGENET1K_V1`, а `INFERENCE_IDX_TO_CLASS` строится из категорий ImageNet. Это согласованный сценарий для чекпоинта и инференса.

## 3) `tests/` + окружение

- **Проблема воспроизводимости тестового окружения сохраняется.**  
  На чистом окружении тесты не стартуют даже на этапе collection из-за отсутствующих зависимостей (`pydantic`, `fastapi`). Для портфолио проект должен подниматься одной командой в venv/docker и проходить хотя бы smoke-набор.
  **Где проверено командой:** `pytest -q`.
