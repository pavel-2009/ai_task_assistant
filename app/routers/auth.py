"""
Роутер для аутентификации и авторизации пользователей.
"""

from fastapi import APIRouter, status, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import UserCreate, User, UserGet
from app.db import get_async_session
from app.auth import hash_password, verify_password, create_access_token


router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)



@router.post("/register", status_code=status.HTTP_201_CREATED, description="Регистрация нового пользователя")
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


@router.post("/login", status_code=status.HTTP_200_OK, description="Аутентификация пользователя")
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