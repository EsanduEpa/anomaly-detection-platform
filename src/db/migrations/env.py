from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context

import sys
import os

# Allow Python to find our src folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import settings to read DATABASE_URL from .env
from src.config import settings

# Import Base and all models so Alembic can see our tables
from src.db.session import Base
from src.models.metric import MetricDataPoint
from src.models.alert import Alert

# Alembic config object
config = context.config

# Set up logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This tells Alembic which tables to track
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live database connection."""
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live database connection."""
    connectable = create_engine(settings.DATABASE_URL)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()


# Alembic decides which function to call based on the situation
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()