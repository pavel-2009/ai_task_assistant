"""
Точка входа в приложение
"""

from fastapi import FastAPI, status, Path, HTTPException, Depends, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, update, delete
import jwt
from jwt.exceptions import InvalidTokenError

import typing
import os
import uuid
from dotenv import load_dotenv

from .models import TaskGet, TaskCreate, TaskUpdate, Task, User, UserCreate, UserGet, UserBase
from .auth import hash_password, verify_password, create_access_token
from .db import get_async_session
from .utils.image_ops import validate_image, resize_image

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")


# Создание приложения
app = FastAPI()

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

    return {"filepath": f"avatars/{filename}", "filename": filename}


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