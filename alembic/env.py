"""Alembic environment.

Reads DATABASE_URL from the environment rather than alembic.ini so the same
migration runs in docker-compose, CI, and production without editing a file
that is checked into git.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Importing the package loads .env, so `alembic upgrade head` needs no shell
# exports. Importing the catalogue module registers its tables on the shared
# metadata -- without it, autogenerate would propose dropping them.
import interior_ai  # noqa: F401 -- side effect: loads .env
from interior_ai.db import catalogue as _catalogue_models  # noqa: F401
from interior_ai.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_url = os.getenv("DATABASE_URL")
if _url:
    if _url.startswith("postgres://"):
        _url = _url.replace("postgres://", "postgresql+psycopg://", 1)
    elif _url.startswith("postgresql://"):
        _url = _url.replace("postgresql://", "postgresql+psycopg://", 1)
    if _url.startswith("postgresql+psycopg://") and "sslmode=" not in _url:
        _url += ("&" if "?" in _url else "?") + "sslmode=require"
    config.set_main_option("sqlalchemy.url", _url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()