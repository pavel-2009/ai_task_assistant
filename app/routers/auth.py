"""Authentication router."""

from fastapi import APIRouter, Depends, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_async_session
from app.core.rate_limit import limiter
from app.schemas import TokenResponse, UserCreate, UserGet
from app.services import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
auth_service = AuthService()


@limiter.limit("5/minute")
@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserGet)
async def register_new_user(
    request: Request,
    user_payload: UserCreate,
    session: AsyncSession = Depends(get_async_session),
):
    return await auth_service.register_user(session=session, user_payload=user_payload)


@router.post("/login", status_code=status.HTTP_200_OK, response_model=TokenResponse)
@limiter.limit("10/minute")
async def login_user(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_session),
):
    return await auth_service.login_user(
        session=session,
        username=form_data.username,
        password=form_data.password,
    )
