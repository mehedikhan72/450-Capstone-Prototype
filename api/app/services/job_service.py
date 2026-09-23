import os
import shutil
import uuid
from typing import Optional

from fastapi import UploadFile

from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.job import Job

INPUT_DIR = os.environ.get("INPUT_DIR", "/app/input")

# flow_rate is a MEASUREMENT, not a selector. It used to be restricted to
# {0, 1000, 5000}, which rejected any real rate an extractor would report — a
# measured 3,200 flows/s returned HTTP 400 even though the engine maps it to FDM
# correctly. Any non-negative rate is now accepted; None still means "all modes".
DEFAULT_THRESHOLD_HIGH = int(os.environ.get("FLOW_THRESHOLD_HIGH", 1000))
DEFAULT_THRESHOLD_EXTREME = int(os.environ.get("FLOW_THRESHOLD_EXTREME", 5000))


class InvalidJobRequest(Exception):
    pass


def create_job(
    file: UploadFile,
    flow_rate: Optional[int],
    label_col: Optional[str],
    benign_label: Optional[str],
    flow_threshold_high: Optional[int] = None,
    flow_threshold_extreme: Optional[int] = None,
) -> dict:
    if flow_rate is not None and flow_rate < 0:
        raise InvalidJobRequest("flow_rate must be a non-negative integer (flows/s), "
                                "or omitted to run all three modes")

    # Validate the rule here rather than in the worker: a bad rule should fail the
    # request, not a job the caller then has to poll to discover has failed.
    high = DEFAULT_THRESHOLD_HIGH if flow_threshold_high is None else flow_threshold_high
    extreme = (DEFAULT_THRESHOLD_EXTREME if flow_threshold_extreme is None
               else flow_threshold_extreme)
    if high < 0 or extreme < 0:
        raise InvalidJobRequest("flow thresholds must be non-negative")
    if extreme <= high:
        raise InvalidJobRequest(
            f"flow_threshold_extreme ({extreme}) must exceed flow_threshold_high ({high})"
        )

    if not file.filename.lower().endswith(".csv"):
        raise InvalidJobRequest("file must be a .csv")

    job_id = str(uuid.uuid4())
    job_input_dir = os.path.join(INPUT_DIR, "jobs")
    os.makedirs(job_input_dir, exist_ok=True)
    input_path = os.path.join(job_input_dir, f"{job_id}.csv")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    session = SessionLocal()
    try:
        job = Job(
            id=job_id,
            status="pending",
            flow_rate=flow_rate,
            label_col=(label_col or None),
            benign_label=(benign_label or "Benign"),
            input_filename=file.filename,
            input_path=input_path,
            flow_threshold_high=flow_threshold_high,
            flow_threshold_extreme=flow_threshold_extreme,
        )
        session.add(job)
        session.commit()
    finally:
        session.close()

    celery_app.send_task("run_prediction_job", args=[job_id])
    return {
        "job_id": job_id,
        "status": "pending",
        "flow_rate": flow_rate,
        "thresholds": {"high": high, "extreme": extreme},
    }


def get_job(job_id: str) -> Optional[Job]:
    session = SessionLocal()
    try:
        return session.get(Job, job_id)
    finally:
        session.close()
