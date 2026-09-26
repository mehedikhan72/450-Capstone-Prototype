import hashlib
import glob
import os
import shutil
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.feedback import Annotation
from app.models.lifecycle import (
    DatasetSnapshot, DeploymentEvent, DeploymentState, ModelVersion, TrainingRun,
)


BASE_DATASET_PATH = os.environ.get("BASE_DATASET_PATH", "/data/DDoS_Dataset.zip")


def start_training() -> dict:
    if not os.path.isfile(BASE_DATASET_PATH):
        raise ValueError(f"base dataset not found: {BASE_DATASET_PATH}")
    session = SessionLocal()
    try:
        if session.scalar(select(Annotation.id).limit(1)) is None:
            raise ValueError("review at least one prediction before training")
        running = session.scalar(select(TrainingRun.id).where(
            TrainingRun.status.in_(("queued", "building", "training", "validating", "reloading"))
        ).limit(1))
        if running:
            raise ValueError(f"training run already active: {running}")
        cutoff = datetime.now(timezone.utc)
        digest = _sha256(BASE_DATASET_PATH)
        snapshot = DatasetSnapshot(base_sha256=digest, annotation_cutoff=cutoff)
        session.add(snapshot)
        session.flush()
        state = session.get(DeploymentState, 1)
        target = "green" if state.active_slot == "blue" else "blue"
        run = TrainingRun(
            snapshot_id=snapshot.id, parent_model_id=state.current_model_id,
            target_slot=target,
        )
        session.add(run)
        session.commit()
        run_id = run.id
    finally:
        session.close()
    celery_app.send_task("train_candidate", args=[run_id], queue="training")
    return {"run_id": run_id, "status": "queued"}


def ensure_bootstrap_model():
    with SessionLocal() as session:
        state = session.get(DeploymentState, 1)
        if state.current_model_id: return
        bundles = glob.glob("/app/detection-engine/weights/*.joblib")
        if len(bundles) != 1: return
        digest = _sha256(bundles[0]); model_id = f"bootstrap-{digest[:12]}"
        model_dir = os.path.join(os.environ.get("MODEL_REGISTRY_DIR", "/registry"), "models", model_id)
        slot_dir = os.path.join(os.environ.get("MODEL_REGISTRY_DIR", "/registry"), "slots", "blue")
        os.makedirs(model_dir, exist_ok=True); os.makedirs(slot_dir, exist_ok=True)
        artifact = os.path.join(model_dir, "model.joblib")
        if not os.path.isfile(artifact): shutil.copy2(bundles[0], artifact)
        if not os.path.isfile(os.path.join(slot_dir, "model.joblib")):
            shutil.copy2(bundles[0], os.path.join(slot_dir, "model.joblib"))
        if session.get(ModelVersion, model_id) is None:
            session.add(ModelVersion(id=model_id, slot="blue", status="current",
                                     artifact_path=artifact, artifact_sha256=digest))
        state.current_model_id = model_id; state.active_slot = "blue"; session.commit()


def mark_ready(model_id: str):
    with SessionLocal() as session:
        model = session.get(ModelVersion, model_id)
        if model:
            model.status = "candidate"
            run = session.get(TrainingRun, model.training_run_id)
            if run: run.status = "ready"
            session.commit()


def promote(model_id: str, actor: str) -> dict:
    with SessionLocal() as session:
        model = session.get(ModelVersion, model_id)
        if not model or model.status != "candidate":
            raise ValueError("model is not a ready candidate")
        state = session.get(DeploymentState, 1)
        if model.parent_id != state.current_model_id:
            raise ValueError("candidate is stale; train it again from the current model")
        previous = state.current_model_id
        if previous:
            old = session.get(ModelVersion, previous)
            if old: old.status = "fallback"
        if state.fallback_model_id and state.fallback_model_id != previous:
            older = session.get(ModelVersion, state.fallback_model_id)
            if older: older.status = "archived"
        state.fallback_model_id = previous
        state.current_model_id = model.id
        state.active_slot = model.slot
        model.status = "current"
        session.add(DeploymentEvent(
            action="promote", model_id=model.id, previous_model_id=previous, actor=actor,
        ))
        session.commit()
        return status(session)


def rollback(actor: str) -> dict:
    with SessionLocal() as session:
        state = session.get(DeploymentState, 1)
        fallback = session.get(ModelVersion, state.fallback_model_id) if state.fallback_model_id else None
        if not fallback:
            raise ValueError("no fallback model available")
        fallback.status = "rollback_pending"
        session.commit()
        celery_app.send_task("reload_fallback", args=[fallback.id, fallback.artifact_path, actor],
                             queue=f"inference_{fallback.slot}")
        return {"status": "rollback_pending", "model_id": fallback.id}


def complete_rollback(model_id: str, actor: str):
    with SessionLocal() as session:
        state = session.get(DeploymentState, 1)
        fallback = session.get(ModelVersion, model_id)
        if not fallback: return
        previous = state.current_model_id
        current = session.get(ModelVersion, previous) if previous else None
        if current: current.status = "failed"
        for candidate in session.scalars(select(ModelVersion).where(
            ModelVersion.slot == fallback.slot,
            ModelVersion.id != fallback.id,
            ModelVersion.status.in_(("candidate", "reloading")),
        )):
            candidate.status = "rejected"
        fallback.status = "current"
        state.current_model_id, state.fallback_model_id = fallback.id, None
        state.active_slot = fallback.slot
        session.add(DeploymentEvent(action="rollback", model_id=fallback.id,
                                    previous_model_id=previous, actor=actor))
        session.commit()


def status(session=None) -> dict:
    own = session is None
    session = session or SessionLocal()
    try:
        state = session.get(DeploymentState, 1)
        runs = list(session.scalars(select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(10)))
        models = list(session.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(10)))
        return {
            "deployment": {"active_slot": state.active_slot, "current": state.current_model_id,
                           "fallback": state.fallback_model_id},
            "runs": [{"id": r.id, "status": r.status, "snapshot_id": r.snapshot_id,
                      "target_slot": r.target_slot, "metrics": r.metrics,
                      "error": r.error_message, "created_at": r.created_at.isoformat()} for r in runs],
            "models": [{"id": m.id, "status": m.status, "slot": m.slot,
                        "metrics": m.metrics, "sha256": m.artifact_sha256,
                        "created_at": m.created_at.isoformat()} for m in models],
        }
    finally:
        if own: session.close()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
