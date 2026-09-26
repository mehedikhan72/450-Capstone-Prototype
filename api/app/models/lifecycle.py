import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text

from app.db.session import Base


now = lambda: datetime.now(timezone.utc)


class DeploymentState(Base):
    __tablename__ = "deployment_state"
    id = Column(Integer, primary_key=True, default=1)
    active_slot = Column(String, nullable=False, default="blue")
    current_model_id = Column(String, nullable=True)
    fallback_model_id = Column(String, nullable=True)
    updated_at = Column(DateTime, default=now, onupdate=now)


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    base_sha256 = Column(String, nullable=False)
    annotation_cutoff = Column(DateTime, nullable=False)
    feedback_rows = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="pending")
    path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    snapshot_id = Column(String, ForeignKey("dataset_snapshots.id"), nullable=False)
    parent_model_id = Column(String, nullable=True)
    target_slot = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now)
    updated_at = Column(DateTime, default=now, onupdate=now)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(String, primary_key=True)
    parent_id = Column(String, nullable=True)
    training_run_id = Column(String, ForeignKey("training_runs.id"), nullable=True)
    slot = Column(String, nullable=False)
    status = Column(String, nullable=False)
    artifact_path = Column(String, nullable=False)
    artifact_sha256 = Column(String, nullable=False)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=now)


class DeploymentEvent(Base):
    __tablename__ = "deployment_events"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    action = Column(String, nullable=False)
    model_id = Column(String, nullable=True)
    previous_model_id = Column(String, nullable=True)
    actor = Column(String, nullable=False)
    created_at = Column(DateTime, default=now)
