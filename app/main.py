"""
Точка входа в приложение
"""

from fastapi import FastAPI, status, Path, HTTPException

import typing

from .models import TaskGet, TaskCreate, TaskUpdate


# Создание приложения
app = FastAPI()


# Имитация БД
database: typing.List[dict] = []
current_task_id: int = 0


# === ЭНДПОИНТЫ ===
@app.get("/ping", status_code=status.HTTP_200_OK, description="Health-check эндпоинт")
async def ping():
    """Health-check"""

    return {
        "status": "OK"
    }


@app.get("/tasks", status_code=status.HTTP_200_OK, description="Получение всех задач")
async def get_tasks():
    """Получение всех задач"""

    return [TaskGet(**task) for task in database]


@app.post("/tasks", status_code=status.HTTP_201_CREATED, description="Создание задачи")
async def create_task(
    task: TaskCreate
):
    """Создание задачи"""

    global current_task_id
    
    task = task.model_dump()
    task['id'] = current_task_id

    database.append(task)

    current_task_id += 1

    return TaskGet(
        **task
    )


@app.get("/tasks/{task_id}", status_code=status.HTTP_200_OK, description="Получение задачи по ID")
async def get_task(
    task_id: int = Path(...)
):
    """Получение задачи по ID"""

    for task in database:
        if task.get('id', None) == task_id:
            return TaskGet(
                **task
            )

    raise HTTPException(
        status_code=404,
        detail="Задача с указанным ID не найдена"
    )


@app.put("/tasks/{task_id}", status_code=status.HTTP_200_OK, description="Обновление задачи")
async def update_task(
    task_id: int = Path(...),
    task_update: typing.Optional[TaskUpdate] = None
):
    """Обновление задачи"""

    for task in database:
        if task.get('id', None) == task_id:
            for field, val in task_update.model_dump().items():
                task[field] = val

            return TaskGet(
                **task
            )

    raise HTTPException(
        status_code=404,
        detail="Задача с указанным ID не найдена"
    )


@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, description="Удаление задачи")
async def delete_task(
    task_id: int = Path(...)
):
    """Удаление задачи"""

    for idx, task in enumerate(database):
        if task.get('id', None) == task_id:
            database.pop(idx)
            return None

    raise HTTPException(
        status_code=404,
        detail="Задача с указанным ID не найдена"
    )
