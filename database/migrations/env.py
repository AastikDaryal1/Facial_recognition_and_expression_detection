"""
database/migrations/env.py
──────────────────────────
Alembic environment configuration.

This file is run by Alembic every time you invoke an alembic command.
It sets up the SQLAlchemy connection and points Alembic at our ORM models
so that autogenerate can detect schema changes automatically.

Key responsibilities:
  1. Add project root to sys.path so imports work regardless of CWD.
  2. Import `settings` to get the DATABASE_URL from .env.
  3. Import `Base` and all ORM models so Alembic sees every table.
  4. Configure the sync connection URL (strip +asyncpg → use psycopg2).
  5. Provide run_migrations_offline() and run_migrations_online() functions
     that Alembic calls depending on whether a live DB connection is available.
"""

import sys
import os

# ── Step 1: Make sure project root is importable ──────────────────────────────
# This file lives at database/migrations/env.py
# We go up three levels: env.py → migrations/ → database/ → project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, text
from alembic import context

# ── Step 2: Import settings (pydantic-settings object) ───────────────────────
from config.settings import settings

# ── Step 3: Import Base + all ORM models ─────────────────────────────────────
# Importing the models module registers all Table objects on Base.metadata.
# Alembic reads Base.metadata to know what tables should exist.
from database.session import Base
from database import models  # noqa: F401 — side-effect import registers all tables

# ── Step 4: Alembic Config object ─────────────────────────────────────────────
config = context.config

# Override the sqlalchemy.url from alembic.ini with our real DATABASE_URL.
# We strip "+asyncpg" because Alembic uses a synchronous psycopg2 driver,
# not the async asyncpg driver used by the running application.
sync_url = settings.database_url.replace("+asyncpg", "")
config.set_main_option("sqlalchemy.url", sync_url)

# ── Step 5: Wire up Python logging from alembic.ini ──────────────────────────
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Step 6: Tell Alembic which metadata to diff against ───────────────────────
# This is the key line — Alembic compares Base.metadata (our ORM definitions)
# against the live database schema to generate migration files automatically.
target_metadata = Base.metadata


# =============================================================================
# Migration runners
# =============================================================================

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Offline mode generates SQL scripts without connecting to the database.
    Useful for generating SQL to review or run manually (e.g. on production).

    Usage:
        alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schemas (e.g. public) in comparisons
        include_schemas=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Online mode connects to the live database and applies migrations directly.
    This is what runs when you do: alembic upgrade head

    We use a synchronous engine here (psycopg2) even though the app uses
    asyncpg at runtime — Alembic does not support async engines natively.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,   # no connection pooling for migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Compare server defaults and column types precisely
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ── Entry point ───────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()