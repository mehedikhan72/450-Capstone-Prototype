import os
import shutil
import uuid
from typing import Optional

from fastapi import UploadFile

from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.job import Job

INPUT_DIR = os.environ.get("INPUT_DIR", "/app/input")
VALID_FLOW_RATES = {0, 1000, 5000}  # None = all modes (HAM+FDM+DFDM); see config.py


class InvalidJobRequest(Exception):
    pass


def create_job(
    file: UploadFile,
    flow_rate: Optional[int],
    label_col: Optional[str],
    benign_label: Optional[str],
) -> dict:
    if flow_rate is not None and flow_rate not in VALID_FLOW_RATES:
        raise InvalidJobRequest(
            f"flow_rate must be one of {sorted(VALID_FLOW_RATES)} or omitted for all modes"
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
        )
        session.add(job)
        session.commit()
    finally:
        session.close()

    celery_app.send_task("run_prediction_job", args=[job_id])
    return {"job_id": job_id, "status": "pending"}


def get_job(job_id: str) -> Optional[Job]:
    session = SessionLocal()
    try:
        return session.get(Job, job_id)
    finally:
        session.close()
