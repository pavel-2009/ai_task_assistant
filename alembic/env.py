from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from app.db import Base
from app.core.config import Settings
# Импортируем все модели, чтобы зарегистрировать их в SQLAlchemy
from app import db_models  # noqa: F401
# Это объект конфигурации Alembic, который предоставляет
# доступ к значениям используемого .ini-файла.
alembic_config = context.config

# Интерпретируем файл конфигурации для логирования Python.
# Эта строка настраивает логгеры.
if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

# Добавьте сюда объект MetaData вашей модели
# для поддержки 'autogenerate'
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# Другие значения из конфигурации, определяемые потребностями env.py,
# можно получить так:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Запустить миграции в режиме 'offline'.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # Преобразуем асинхронный URL в синхронный для Alembic
    settings = Settings()
    url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Запустить миграции в режиме 'online'.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Преобразуем асинхронный URL в синхронный для Alembic
    settings = Settings()
    url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    configuration = alembic_config.get_section(alembic_config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
