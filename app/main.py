"""
Точка входа в приложение
"""

from fastapi import FastAPI, status, Path, HTTPException, Depends, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
import jwt
from jwt.exceptions import InvalidTokenError
from celery.result import AsyncResult

import typing
import os
from pathlib import Path as PathlibPath
import uuid
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging

from .models import TaskGet, TaskCreate, TaskUpdate, Task, User, UserCreate, UserGet
from .auth import hash_password, verify_password, create_access_token
from .db import get_async_session
from .utils.image_ops import validate_image, resize_image
from .ml.inference_service import InferenceService
from .ml.yolo_service import YoloService, MODEL_PATH
from .ml.config import config
from .ml.tasks import detect_and_visualize_task, segment_image_task


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")


IDX_TO_CLASS = {0: "cat", 1: "dog", 2: "house"} 


# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_service
    global yolo_service
    
    # === STARTUP ===
    try:
        logger.info("🚀 Starting app...")
        
        try: 
            logger.info("🚀 Loading ML model from %s", config.output_dir / "model")
            checkpoints_path = config.output_dir / "model.pth"
            ml_service = InferenceService(
                checkpoints_path=checkpoints_path,
                idx_to_class=IDX_TO_CLASS
            )
            logger.info("✅ ML model loaded successfully!")
        
        except Exception as e:
            logger.error("❌ Failed to load ML model: %s", str(e))
            raise
        
        try:
            logger.info("🚀 Loading YOLO model...")
            yolo_service = YoloService(MODEL_PATH)
            logger.info("✅ YOLO model loaded successfully!")
        
        except Exception as e:
            logger.error("❌ Failed to load YOLO model: %s", str(e))
            raise
        
    except Exception as e:
        logger.error("❌ Error during app startup: %s", str(e))
        raise
        
    yield
            
    # === SHUTDOWN ===
    try:
        logger.info("🛑 Shutting down app...")
        
        if ml_service:
            try: 
                del ml_service
                logger.info("✅ ML service cleaned up successfully!")
            
            except Exception as e:
                logger.error("❌ Failed to clean up ML service: %s", str(e))
                raise
            
        if yolo_service:
            try:
                del yolo_service
                logger.info("✅ YOLO service cleaned up successfully!")
            
            except Exception as e:
                logger.error("❌ Failed to clean up YOLO service: %s", str(e))
                raise
            
        logger.info("✅ App shutdown completed successfully!")
        
    except Exception as e:
        logger.error("❌ Error during app shutdown: %s", str(e))
            



# Создание приложения
app = FastAPI(lifespan=lifespan)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_async_session)) -> User:
    """Получение текущего пользователя по JWT токену"""

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("user_id")

        if user_id is None:
            raise InvalidTokenError("Invalid token")

        user = await session.execute(select(User).where(User.id == user_id))
        user = user.scalar_one_or_none()

        if user is None:
            raise InvalidTokenError("User not found")

        return user

    except InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))


# === ЭНДПОИНТЫ ===
@app.get("/ping", status_code=status.HTTP_200_OK, description="Health-check эндпоинт")
async def ping():
    """Health-check"""

    return {
        "status": "OK"
    }


@app.get("/tasks", status_code=status.HTTP_200_OK, description="Получение всех задач")
async def get_tasks(
    session: AsyncSession = Depends(get_async_session)
):
    """Получение всех задач"""

    task = await session.execute(select(Task))
    tasks = task.scalars().all()

    return [
        TaskGet(
            id=task.id,
            title=task.title,
            description=task.description,
            author_id=task.author_id
        )
        for task in tasks
    ]


@app.post("/tasks", status_code=status.HTTP_201_CREATED, description="Создание задачи")
async def create_task(
    task: TaskCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Создание задачи"""
    
    task = Task(**task.model_dump())
    task.author_id = current_user.id

    session.add(task)
    await session.commit()

    await session.refresh(task)

    return task


@app.get("/tasks/{task_id}", status_code=status.HTTP_200_OK, description="Получение задачи по ID")
async def get_task(
    task_id: int = Path(...),
    session: AsyncSession = Depends(get_async_session)
):
    """Получение задачи по ID"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is not None:
        return TaskGet(
            id=task.id,
            title=task.title,
            description=task.description,
            author_id=task.author_id
        )

    raise HTTPException(
        status_code=404,
        detail="Задача с указанным ID не найдена"
    )


@app.put("/tasks/{task_id}", status_code=status.HTTP_200_OK, description="Обновление задачи")
async def update_task(
    task_id: int = Path(...),
    task_update: typing.Optional[TaskUpdate] = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Обновление задачи"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )
    
    if task.author_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="У вас нет прав на изменение этой задачи"
        )

    update_dict = task_update.model_dump(exclude_unset=True) if task_update else {}

    await session.execute(
        update(Task).where(Task.id == task_id).values(**update_dict)
    )
    await session.commit()

    return TaskGet(
        id=task.id,
        title=update_dict.get("title", task.title),
        description=update_dict.get("description", task.description)
    )


@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, description="Удаление задачи")
async def delete_task(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Удаление задачи"""

    task = await session.execute(select(Task).where(Task.id == task_id))
    task = task.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Задача с указанным ID не найдена"
        )

    if task.author_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="У вас нет прав на удаление этой задачи"
        )

    await session.execute(
        delete(Task).where(Task.id == task_id)
    )
    await session.commit()
    
    return None


@app.post("/tasks/{task_id}/avatar", status_code=status.HTTP_200_OK, description="Загрузка аватара для задачи")
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

    return {"filepath": f"avatars/{filename}", "filename": filename}


@app.post("/tasks/{task_id}/predict", status_code=status.HTTP_200_OK, description="Предсказание класса изображения для задачи")
async def predict_img_class(
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
        predictions = ml_service.predict(image_bytes)
        return {"predicted_class": predictions["class_name"], "confidence": predictions["confidence"]}
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
        
@app.post("/tasks/{task_id}/detect/submit", status_code=status.HTTP_202_ACCEPTED, description="Детекция объектов на аватарке задачи")
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
    
    return {"message": "Задача на детекцию объектов успешно создана", "celery_task_id": celery_task.id}


@app.post("/tasks/{task_id}/detect/status/{celery_task_id}", status_code=status.HTTP_200_OK, description="Получение результатов детекции объектов на аватарке задачи")
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
        return {"status": "PENDING", "message": "Задача еще не началась"}
    
    elif celery_task.state == "STARTED":
        return {"status": "STARTED", "message": "Задача выполняется"}
    
    elif celery_task.state == "SUCCESS":
        result = celery_task.result
        return {"status": "SUCCESS", "result": result}
    
    elif celery_task.state == "FAILURE":
        return {"status": "FAILURE", "message": "Задача завершилась с ошибкой"}
    
    else:
        return {"status": celery_task.state, "message": "Задача в неизвестном состоянии"}
    
    
@app.post("/tasks/{task_id}/segment/submit", status_code=status.HTTP_202_ACCEPTED, description="Сегментация аватарки задачи")
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
    
    return {"message": "Задача на сегментацию изображения успешно создана", "celery_task_id": celery_task.id}


@app.get("/tasks/{task_id}/segment/status/{celery_task_id}", status_code=status.HTTP_200_OK, description="Получение результатов сегментации аватарки задачи")
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
        return {"status": "PENDING", "message": "Задача еще не началась"}
    
    elif celery_task.state == "STARTED":
        return {"status": "STARTED", "message": "Задача выполняется"}
    
    elif celery_task.state == "SUCCESS":
        return {"status": "SUCCESS", "result_path": f"segments/{task_id}_segmentation.png"}
    
    elif celery_task.state == "FAILURE":
        return {"status": "FAILURE", "message": "Задача завершилась с ошибкой"}
    
    
@app.get("/tasks/{task_id}/segment/download/", status_code=status.HTTP_200_OK, description="Скачивание результатов сегментации аватарки задачи")
async def download_segmented_image(
    task_id: int = Path(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Скачивание результатов сегментации аватарки задачи"""
    
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
        
    segmented_image_path = PathlibPath("/app/avatars/segments") / f"{task_id}_segmentation.png"
    
    logger.info(f"Looking for segmentation file at: {segmented_image_path}")
    logger.info(f"File exists: {os.path.exists(segmented_image_path)}")
    
    if not os.path.exists(segmented_image_path):
        raise HTTPException(
            status_code=404,
            detail="Результат сегментации не найден. Убедитесь, что задача на сегментацию была успешно выполнена."
        )
        
    return FileResponse(segmented_image_path, media_type="image/png", filename=f"{task_id}_segmentation.png")
    
    

# === ЭНДПОИНТЫ АУТЕНТИФИКАЦИИ ===

@app.post("/auth/register", status_code=status.HTTP_201_CREATED, description="Регистрация нового пользователя")
async def register_new_user(
    user_payload: UserCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """Регистрация нового пользователя"""

    user = await session.execute(select(User).where(User.username == user_payload.username))
    user = user.scalar_one_or_none()

    if user:
        raise HTTPException(
            status_code=400,
            detail="Пользователь с таким именем уже существует"
        )

    hashed_password = hash_password(user_payload.password)

    new_user = User(
        username=user_payload.username,
        password=hashed_password
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    return UserGet(
        id=new_user.id,
        username=new_user.username,
    )


@app.post("/auth/login", status_code=status.HTTP_200_OK, description="Аутентификация пользователя")
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_session)
):
    """Аутентификация пользователя"""

    user = await session.execute(select(User).where(User.username == form_data.username))
    user = user.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=400,
            detail="Неверное имя пользователя или пароль"
        )

    if not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=400,
            detail="Неверное имя пользователя или пароль"
        )

    token = create_access_token(user_id=user.id)

    return {"access_token": token, "token_type": "bearer"}