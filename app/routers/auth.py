"""
Роутер для аутентификации и авторизации пользователей.
"""

from fastapi import APIRouter, status, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db_models import User
from app.schemas import UserCreate, UserGet, TokenResponse
from app.db import get_async_session
from app.auth import create_access_token
from app.core.rate_limit import limiter
from app.core.security import validate_password_strength, hash_password, verify_password


router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)


@limiter.limit("5/minute")  # Ограничение на 5 запросов в минуту для всех эндпоинтов в этом роутере
@router.post("/register", status_code=status.HTTP_201_CREATED, description="Регистрация нового пользователя", response_model=UserGet)
async def register_new_user(
    request: Request,
    user_payload: UserCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """Регистрация нового пользователя"""
    
    if not user_payload.username or not user_payload.password:
        raise HTTPException(
            status_code=400,
            detail="Имя пользователя и пароль не могут быть пустыми"
        )

    user = await session.execute(select(User).where(User.username == user_payload.username))
    user = user.scalar_one_or_none()

    if user:
        raise HTTPException(
            status_code=400,
            detail="Пользователь с таким именем уже существует"
        )
        
    password = user_payload.password
    if not validate_password_strength(password):
        raise HTTPException(
            status_code=400,
            detail="Пароль должен быть не менее 8 символов, содержать заглавные и строчные буквы, цифры и специальные символы"
        )

    hashed_password = hash_password(password)

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


@router.post("/login", status_code=status.HTTP_200_OK, description="Аутентификация пользователя", response_model=TokenResponse)
@limiter.limit("10/minute")  # Ограничение на 10 запросов в минуту для эндпоинта логина
async def login_user(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_session)
):
    """Аутентификация пользователя"""
    
    if not form_data.username or not form_data.password:
        raise HTTPException(
            status_code=400,
            detail="Имя пользователя и пароль не могут быть пустыми"
        )

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

    return TokenResponse(access_token=token, token_type="bearer")