import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://ddos:ddos@localhost:5432/ddos"
)

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


# Columns added after the jobs table already existed in a deployed database.
# create_all() creates missing TABLES but never alters an existing one, so without
# this a redeploy over a live volume would start and then fail on every insert.
# ADD COLUMN IF NOT EXISTS is idempotent, so it is safe on a fresh database too.
_MIGRATIONS = (
    "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS flow_threshold_high INTEGER",
    "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS flow_threshold_extreme INTEGER",
)


def init_db():
    from app.models.job import Job  # noqa: F401  registers the table on Base.metadata

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for stmt in _MIGRATIONS:
            try:
                conn.execute(text(stmt))
            except Exception as e:      # non-fatal: log and continue, never block startup
                print(f"[db] migration skipped ({stmt}): {e}")
