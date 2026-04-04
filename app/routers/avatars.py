"""
Роутер для работы с аватарами пользователей
"""

from fastapi import APIRouter, status, Path, Depends, UploadFile, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from celery.result import AsyncResult

import os
import uuid

from app.db_models import Task, User
from app.db import get_async_session
from app.auth import get_current_user
from app.core import config
from app.utils.image_ops import validate_image, resize_image
from app.ml.cv.tasks import detect_and_visualize_task, segment_image_task, predict_avatar_class
from app.ml.nlp.tasks import update_recommendations_for_task
from app.schemas import (
    FileUploadResponse, CeleryTaskResponse, CeleryTaskStatusResponse
)


router = APIRouter(
    prefix="/tasks",
    tags=["avatars"]
)


@router.post("/{task_id}/avatar", status_code=status.HTTP_200_OK, description="Загрузка аватара для задачи", response_model=FileUploadResponse)
async def upload_avatar(
    task_id: int = Path(...),
    image: UploadFile = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Загрузка аватара для задачи"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Изображение не предоставлено"
        )
    
    if task.author_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="У вас нет прав на загрузку аватара для этой задачи"
        )

    with image.file as f:
        image_bytes = f.read()


    if not validate_image(image_bytes):
        raise HTTPException(
            status_code=400,
            detail="Невалидное изображение"
        )

    image = resize_image(image_bytes)

    if not os.path.exists("avatars"):
        os.makedirs("avatars")  

    filename = f"{task_id}_{uuid.uuid4().hex}.jpeg"

    with open(f"avatars/{filename}", "wb") as f:
        f.write(image)

    await session.execute(
        update(Task).where(Task.id == task_id).values(avatar_file=f"avatars/{filename}")
    )
    await session.commit()
    
    # После загрузки аватара обновляем рекомендации для задачи
    update_recommendations_for_task.delay(task_id=task_id)

    return FileUploadResponse(filepath=f"avatars/{filename}", filename=filename)


@router.post("/{task_id}/predict/submit", status_code=status.HTTP_202_ACCEPTED, description="Создание задачи на предсказание класса аватарки задачи", response_model=CeleryTaskResponse)
async def predict_img_class_submit(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Предсказание класса изображения аватара для задачи"""

    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Требуется аутентификация"
        )

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
    
    if task.avatar_file is None:
        raise HTTPException(
            status_code=400,
            detail="Для этой задачи не загружен аватар"
        )
    
    image_path = task.avatar_file
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Файл аватара не найден"
        )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    try:
        celery_task = predict_avatar_class.delay(task_id=task_id, image_path=image_path)
        return CeleryTaskResponse(message="Задача на предсказание класса аватарки успешно создана", celery_task_id=celery_task.id)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
        
@router.get("/{task_id}/predict/status/{celery_task_id}", status_code=status.HTTP_200_OK, description="Получение результатов предсказания класса аватарки задачи", response_model=CeleryTaskStatusResponse)
async def get_predict_class_results(
    task_id: int = Path(...),
    celery_task_id: str = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Получение результатов предсказания класса аватарки задачи"""
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Требуется аутентификация"
        )
        
    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
        
    if task.avatar_file is None:
        raise HTTPException(
            status_code=400,
            detail="Для этой задачи не загружен аватар"
        )
        
    image_path = task.avatar_file
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Файл аватара не найден"
        )
        
    celery_task = AsyncResult(celery_task_id)
    if not celery_task:
        raise HTTPException(
            status_code=404,
            detail="Задача Celery с указанным ID не найдена"
        )
        
    if celery_task.state == "PENDING":
        return CeleryTaskStatusResponse(status="PENDING", message="Задача еще не началась")
    
    elif celery_task.state == "STARTED":
        return CeleryTaskStatusResponse(status="STARTED", message="Задача выполняется")
    
    elif celery_task.state == "SUCCESS":
        result = celery_task.result
        return CeleryTaskStatusResponse(status="SUCCESS", result=result.get("predicted_class") if result else None)
    
    elif celery_task.state == "FAILURE":
        return CeleryTaskStatusResponse(status="FAILURE", message="Задача завершилась с ошибкой")
    
    return CeleryTaskStatusResponse(status=celery_task.state, message="Неизвестное состояние")
        
        
@router.post("/{task_id}/detect/submit", status_code=status.HTTP_202_ACCEPTED, description="Детекция объектов на аватарке задачи", response_model=CeleryTaskResponse)
async def detect_objects(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Создание фоновой задачи для детекции объектов на аватарке задачи"""
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Требуется аутентификация"
        )
        
    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
        
    if not task.avatar_file:
        raise HTTPException(
            status_code=400,
            detail="Для этой задачи не загружен аватар"
        )
        
    image_path = task.avatar_file
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Файл аватара не найден"
        )
        
    celery_task = detect_and_visualize_task.delay(task_id=task_id, image_path=image_path)
    
    return CeleryTaskResponse(message="Задача на детекцию объектов успешно создана", celery_task_id=celery_task.id)


@router.get("/{task_id}/detect/status/{celery_task_id}", status_code=status.HTTP_200_OK, description="Получение результатов детекции объектов на аватарке задачи", response_model=CeleryTaskStatusResponse)
async def get_detection_results(
    task_id: int = Path(...),
    celery_task_id: str = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Получение результатов детекции объектов на аватарке задачи"""
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Требуется аутентификация"
        )
        
    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
        
    if not task.avatar_file:
        raise HTTPException(
            status_code=400,
            detail="Для этой задачи не загружен аватар"
        )
        
    image_path = task.avatar_file
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Файл аватара не найден"
        )
        
    celery_task = AsyncResult(celery_task_id)
    if not celery_task:
        raise HTTPException(
            status_code=404,
            detail="Задача Celery с указанным ID не найдена"
        )
        
    if celery_task.state == "PENDING":
        return CeleryTaskStatusResponse(status="PENDING", message="Задача еще не началась")
    
    elif celery_task.state == "STARTED":
        return CeleryTaskStatusResponse(status="STARTED", message="Задача выполняется")
    
    elif celery_task.state == "SUCCESS":
        result = celery_task.result
        return CeleryTaskStatusResponse(status="SUCCESS", result=result)
    
    elif celery_task.state == "FAILURE":
        return CeleryTaskStatusResponse(status="FAILURE", message="Задача завершилась с ошибкой")
    
    return CeleryTaskStatusResponse(status=celery_task.state, message="Неизвестное состояние")
    
    
@router.post("/{task_id}/segment/submit", status_code=status.HTTP_202_ACCEPTED, description="Сегментация аватарки задачи", response_model=CeleryTaskResponse)
async def segment_image(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Создание фоновой задачи для сегментации аватарки задачи"""
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Требуется аутентификация"
        )
        
    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
        
    if not task.avatar_file:
        raise HTTPException(
            status_code=400,
            detail="Для этой задачи не загружен аватар"
        )
        
    image_path = task.avatar_file
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Файл аватара не найден"
        )
        
    celery_task = segment_image_task.delay(task_id=task_id, image_path=image_path)
    
    return CeleryTaskResponse(message="Задача на сегментацию изображения успешно создана", celery_task_id=celery_task.id)


@router.get("/{task_id}/segment/status/{celery_task_id}", status_code=status.HTTP_200_OK, description="Получение результатов сегментации аватарки задачи", response_model=CeleryTaskStatusResponse)
async def get_segmentation_results(
    task_id: int = Path(...),
    celery_task_id: str = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Получение результатов сегментации аватарки задачи"""
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Требуется аутентификация"
        )
        
    if not celery_task_id:
        raise HTTPException(
            status_code=400,
            detail="ID задачи Celery не предоставлен"
        )
        
    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
        
    if not task.avatar_file:
        raise HTTPException(
            status_code=400,
            detail="Для этой задачи не загружен аватар"
        )
        
    image_path = task.avatar_file
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404,
            detail="Файл аватара не найден"
        )
        
    celery_task = AsyncResult(celery_task_id)
    
    if not celery_task:
        raise HTTPException(
            status_code=404,
            detail="Задача Celery с указанным ID не найдена"
        )
        
    if celery_task.state == "PENDING":
        return CeleryTaskStatusResponse(status="PENDING", message="Задача еще не началась")
    
    elif celery_task.state == "STARTED":
        return CeleryTaskStatusResponse(status="STARTED", message="Задача выполняется")
    
    elif celery_task.state == "SUCCESS":
        return CeleryTaskStatusResponse(status="SUCCESS", result=f"segments/{task_id}_segmentation.png")
    
    elif celery_task.state == "FAILURE":
        return CeleryTaskStatusResponse(status="FAILURE", message="Задача завершилась с ошибкой")
    
    return CeleryTaskStatusResponse(status=celery_task.state, message="Неизвестное состояние")
    
    
@router.get("/{task_id}/segment/download/", status_code=status.HTTP_200_OK, description="Скачивание результатов сегментации аватарки задачи (возвращает image/png)")
async def download_segmented_image(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Скачивание результатов сегментации аватарки задачи.
    Возвращает изображение в формате PNG.
    """
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Требуется аутентификация"
        )
      
    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
        
    segmented_image_path = config.avatars_segments_dir / f"{task_id}_segmentation.png"
    
    if not os.path.exists(segmented_image_path):
        raise HTTPException(
            status_code=404,
            detail="Результат сегментации не найден. Убедитесь, что задача на сегментацию была успешно выполнена."
        )
        
    return FileResponse(segmented_image_path, media_type="image/png", filename=f"{task_id}_segmentation.png")
    
