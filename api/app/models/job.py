import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db.session import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, nullable=False, default="pending")  # pending|running|done|failed
    flow_rate = Column(Integer, nullable=True)  # MEASURED flows/s; None = run all three modes
    # The switching rule this job ran under. NULL means "server defaults applied".
    # Recorded per job because the ns-3 control frontend sets them per run, so a mode
    # cannot be reconstructed from the rate alone.
    flow_threshold_high = Column(Integer, nullable=True)
    flow_threshold_extreme = Column(Integer, nullable=True)
    label_col = Column(String, nullable=True)
    benign_label = Column(String, nullable=True)
    input_filename = Column(String, nullable=False)
    input_path = Column(String, nullable=False)
    output_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
