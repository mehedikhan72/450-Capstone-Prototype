import os
import shutil
import sys
import traceback

import inference as inf
import models as _models

from app.db.session import SessionLocal, init_db
from app.models.job import Job

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")


def _bind_main_module_classes():
    """
    The weight pickles were created while `inference.py` ran as `__main__`
    (its `from models import VMFCVD, ...` line binds those names onto
    __main__). Unpickling elsewhere needs the same names on whatever module
    is __main__ at the time — here, celery's entrypoint. Replicate that
    binding so pickle.load can resolve `__main__.VMFCVD` etc.
    """
    main = sys.modules["__main__"]
    for name in (
        "VMFCVD", "FastDetectionMode", "DefensiveFastDetectionMode",
        "HighAccuracyMode", "VMFCVDVoter", "DetailedResourceMonitor",
    ):
        setattr(main, name, getattr(_models, name))


def load_model():
    """Load pickled model artifacts once per worker process (expensive)."""
    init_db()
    _bind_main_module_classes()
    inf.load_artifacts()


def run_prediction(job_id: str):
    session = SessionLocal()
    job = session.get(Job, job_id)
    if job is None:
        session.close()
        return

    job.status = "running"
    session.commit()

    try:
        inf.run_csv(
            csv_path=job.input_path,
            flow_rate=job.flow_rate,
            label_col=job.label_col,
            benign_label=job.benign_label or "Benign",
            save_output=True,
            track_resources=False,
        )

        job_out_dir = os.path.join(OUTPUT_DIR, "jobs", job_id)
        os.makedirs(job_out_dir, exist_ok=True)
        dest = os.path.join(job_out_dir, "predictions.csv")
        shutil.move(inf.cfg.OUT_PATH, dest)

        job.output_path = dest
        job.status = "done"
        job.error_message = None
    except Exception as e:
        job.status = "failed"
        job.error_message = f"{e}\n{traceback.format_exc()[-2000:]}"
    finally:
        session.commit()
        session.close()
