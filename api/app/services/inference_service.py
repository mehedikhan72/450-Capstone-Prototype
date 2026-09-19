import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

import inference as inf
import models as _models

from app.db.session import SessionLocal, init_db
from app.models.job import Job
from app.services import log_service

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


def _prediction_summary(p_bin) -> dict:
    p_bin = np.asarray(p_bin)
    n = len(p_bin)
    n_mal = int((p_bin == 1).sum())
    return {
        "total": n,
        "benign": n - n_mal,
        "malicious": n_mal,
        "malicious_pct": round(n_mal / n * 100, 2) if n else 0.0,
    }


def run_prediction(job_id: str):
    session = SessionLocal()
    job = session.get(Job, job_id)
    if job is None:
        session.close()
        return

    job.status = "running"
    session.commit()

    t0 = time.perf_counter()
    log_entry = {
        "job_id": job_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_filename": job.input_filename,
        "flow_rate": job.flow_rate,
    }

    try:
        all_res, df_out = inf.run_csv(
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

        log_entry["status"] = "done"
        log_entry["rows"] = len(df_out)
        log_entry["modes_run"] = list(all_res.keys())
        log_entry["results"] = {
            mode: {
                "predictions": _prediction_summary(res["predictions_binary"]),
                "metrics_binary": res.get("metrics_binary"),
                "metrics_3label": res.get("metrics_3label"),
            }
            for mode, res in all_res.items()
        }
    except Exception as e:
        job.status = "failed"
        job.error_message = f"{e}\n{traceback.format_exc()[-2000:]}"

        log_entry["status"] = "failed"
        log_entry["error"] = str(e)
    finally:
        log_entry["duration_seconds"] = round(time.perf_counter() - t0, 4)
        session.commit()
        session.close()
        log_service.append_entry(log_entry)
