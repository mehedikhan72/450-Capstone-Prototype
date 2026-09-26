import hashlib
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
from app.models.lifecycle import ModelVersion
from app.services import feedback_service, log_service

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
MODEL_VERSION = None


def _bind_main_module_classes():
    """
    The TRIDENT v13 bundle was pickled from a notebook, where the classes were
    defined at top level and therefore live on `__main__`. Unpickling elsewhere
    needs the same names on whatever module is __main__ at the time — here,
    celery's entrypoint. Replicate that binding so joblib.load can resolve
    `__main__.Trident`, `__main__.Member` and friends.

    These five are every class reachable from the bundle: Pre holds a
    LinRegImputer, each Trident holds Members, and DFDM holds a Trident.
    """
    main = sys.modules["__main__"]
    for name in (
        "LinRegImputer", "Pre", "Member", "Trident", "DFDM",
        "DetailedResourceMonitor",
    ):
        setattr(main, name, getattr(_models, name))


def load_model(ckpt_path=None):
    """Load pickled model artifacts once per worker process (expensive)."""
    init_db()
    _bind_main_module_classes()
    global MODEL_VERSION
    slot = os.environ.get("MODEL_SLOT")
    slot_path = os.path.join(os.environ.get("MODEL_REGISTRY_DIR", "/registry"),
                             "slots", slot or "", "model.joblib")
    source = (ckpt_path or (slot_path if slot and os.path.isfile(slot_path) else None)
              or os.environ.get("MODEL_BUNDLE_PATH") or inf.cfg.CKPT_DIR)
    artifacts = inf.load_artifacts(source)
    bundle_path = inf._find_bundle(source)
    with open(bundle_path, "rb") as bundle:
        full_digest = hashlib.sha256(bundle.read()).hexdigest()
    digest = full_digest[:12]
    schema = artifacts.get("schema") or {}
    with SessionLocal() as session:
        registered = session.query(ModelVersion).filter_by(artifact_sha256=full_digest).first()
    MODEL_VERSION = registered.id if registered else f"bootstrap-{digest}"
    print(f"[model] version={MODEL_VERSION}")


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


def _model_breakdown(res: dict) -> dict:
    preds_by_model = res.get("model_predictions") or {}
    metrics_by_model = res.get("model_metrics") or {}
    report_by_model = res.get("model_report") or {}
    return {
        name: {
            "predictions": _prediction_summary(preds),
            "metrics_binary": metrics_by_model.get(name),
            # v13's member_report: Shapley weight, per-sample cost and, ONLY when
            # the upload carried a Label, the member's own MCC/recall/precision.
            # With no Label the metric keys are absent rather than zero-filled.
            "member_report": report_by_model.get(name),
        }
        for name, preds in preds_by_model.items()
    }


def run_prediction(job_id: str):
    session = SessionLocal()
    job = session.get(Job, job_id)
    if job is None:
        session.close()
        return

    job.status = "running"
    job.model_version = MODEL_VERSION
    session.commit()

    t0 = time.perf_counter()
    log_entry = {
        "job_id": job_id,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_filename": job.input_filename,
        "flow_rate": job.flow_rate,
    }

    try:
        thresholds = inf.resolve_thresholds(
            job.flow_threshold_high, job.flow_threshold_extreme
        )
        log_entry["thresholds"] = thresholds
        log_entry["mode_resolved"] = inf._flow_to_mode(job.flow_rate, thresholds) or "ALL"
        trained = (inf.SCHEMA or {}).get("flow_thresholds") or {}
        if trained and (trained.get("high"), trained.get("extreme")) != (
            thresholds["high"],
            thresholds["extreme"],
        ):
            # Surfaced in /logs/recent so a scaled demonstration is never mistaken
            # for a run on the deployment contract the budgets were derived from.
            log_entry["thresholds_trained"] = trained
            log_entry["thresholds_overridden"] = True

        all_res, df_out = inf.run_csv(
            csv_path=job.input_path,
            flow_rate=job.flow_rate,
            label_col=job.label_col,
            benign_label=job.benign_label or "Benign",
            save_output=True,
            track_resources=False,
            flow_threshold_high=job.flow_threshold_high,
            flow_threshold_extreme=job.flow_threshold_extreme,
        )
        df_out["model_version"] = MODEL_VERSION
        df_out.to_csv(inf.cfg.OUT_PATH, index=False)

        job_out_dir = os.path.join(OUTPUT_DIR, "jobs", job_id)
        os.makedirs(job_out_dir, exist_ok=True)
        dest = os.path.join(job_out_dir, "predictions.csv")
        shutil.move(inf.cfg.OUT_PATH, dest)

        job.output_path = dest
        job.status = "done"
        job.error_message = None

        feedback_service.persist_predictions(job_id, MODEL_VERSION, df_out)

        log_entry["status"] = "done"
        log_entry["rows"] = len(df_out)
        log_entry["modes_run"] = list(all_res.keys())
        log_entry["results"] = {
            mode: {
                "predictions": _prediction_summary(res["predictions_binary"]),
                "metrics_binary": res.get("metrics_binary"),
                "metrics_3label": res.get("metrics_3label"),
                "models": _model_breakdown(res),
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
