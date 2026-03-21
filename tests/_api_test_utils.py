from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
import asyncio
import os
import shutil
import sys
import tempfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

ROOT_DIR = Path(__file__).resolve().parents[1]

os.environ.setdefault("SECRET_KEY", "test-secret-key")

stub_cv_tasks = ModuleType("app.ml.cv.tasks")

def _delay_stub(*args, **kwargs):
    return SimpleNamespace(id="stub-task-id")

stub_cv_tasks.predict_avatar_class = SimpleNamespace(delay=_delay_stub)
stub_cv_tasks.detect_and_visualize_task = SimpleNamespace(delay=_delay_stub)
stub_cv_tasks.segment_image_task = SimpleNamespace(delay=_delay_stub)
sys.modules.setdefault("app.ml.cv.tasks", stub_cv_tasks)

from app.db import Base, get_async_session
import app.auth as auth_module
from app.routers import auth, avatars, tasks


class FakeNerService:
    def tag_task(self, text: str) -> dict:
        return {"technologies": [], "confidence": 0}


@contextmanager
def make_test_client():
    temp_dir = tempfile.TemporaryDirectory()
    db_path = Path(temp_dir.name) / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(tasks.router)
    app.include_router(avatars.router)

    async def override_get_async_session():
        async with session_factory() as session:
            yield session

    @asynccontextmanager
    async def test_lifespan(_app):
        _app.state.ner_service = FakeNerService()
        yield

    async def setup_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def teardown_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    asyncio.run(setup_db())
    auth_module.SECRET_KEY = "test-secret-key"
    app.dependency_overrides[get_async_session] = override_get_async_session
    app.router.lifespan_context = test_lifespan
    app.state.ner_service = FakeNerService()

    original_validate = avatars.validate_image
    original_resize = avatars.resize_image
    avatars.validate_image = lambda image_bytes: True
    avatars.resize_image = lambda image_bytes, max_size=1024: image_bytes

    avatars_dir = ROOT_DIR / "avatars"
    if avatars_dir.exists():
        shutil.rmtree(avatars_dir)

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        avatars.validate_image = original_validate
        avatars.resize_image = original_resize
        if avatars_dir.exists():
            shutil.rmtree(avatars_dir)
        asyncio.run(teardown_db())
        temp_dir.cleanup()
