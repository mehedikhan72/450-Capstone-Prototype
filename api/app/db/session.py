import os

from sqlalchemy import create_engine, inspect, text
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
_JOB_COLUMNS = {
    "flow_threshold_high": "INTEGER",
    "flow_threshold_extreme": "INTEGER",
    "model_version": "VARCHAR",
    "worker_slot": "VARCHAR",
}


def init_db():
    from app.models.job import Job  # noqa: F401  registers the table on Base.metadata
    from app.models.feedback import Annotation, Prediction  # noqa: F401
    from app.models.lifecycle import (  # noqa: F401
        DatasetSnapshot, DeploymentEvent, DeploymentState, ModelVersion, TrainingRun,
    )

    Base.metadata.create_all(engine)
    with SessionLocal() as session:
        from app.models.lifecycle import DeploymentState
        if session.get(DeploymentState, 1) is None:
            session.add(DeploymentState(id=1, active_slot="blue"))
            session.commit()
    existing = {column["name"] for column in inspect(engine).get_columns("jobs")}
    if_not_exists = "IF NOT EXISTS " if engine.dialect.name == "postgresql" else ""
    with engine.begin() as conn:
        for name, column_type in _JOB_COLUMNS.items():
            if name not in existing:
                conn.execute(text(
                    f"ALTER TABLE jobs ADD COLUMN {if_not_exists}{name} {column_type}"
                ))
