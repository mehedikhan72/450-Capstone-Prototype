import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Text, UniqueConstraint

from app.db.session import Base


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("job_id", "source_row_key"),)

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True)
    source_row_key = Column(String, nullable=False)
    flow_id = Column(String, nullable=True, index=True)
    model_version = Column(String, nullable=False, index=True)
    original_label = Column(String, nullable=True)
    outputs = Column(JSON, nullable=False)
    row_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id = Column(
        String, ForeignKey("predictions.id"), nullable=False, index=True
    )
    label = Column(String, nullable=False)
    reviewer = Column(String, nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
