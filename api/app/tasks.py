import os
from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.services import inference_service


@worker_process_init.connect
def _load_model(**kwargs):
    if os.environ.get("WORKER_ROLE", "inference") == "inference":
        inference_service.load_model()


@celery_app.task(name="run_prediction_job")
def run_prediction_job(job_id: str):
    inference_service.run_prediction(job_id)


@celery_app.task(name="reload_model")
def reload_model(model_id: str, artifact_path: str):
    inference_service.load_model(artifact_path)
    from app.services.lifecycle_service import mark_ready
    mark_ready(model_id)


@celery_app.task(name="reload_fallback")
def reload_fallback(model_id: str, artifact_path: str, actor: str):
    inference_service.load_model(artifact_path)
    from app.services.lifecycle_service import complete_rollback
    complete_rollback(model_id, actor)


@celery_app.task(name="train_candidate")
def train_candidate(run_id: str):
    from app.services.training_service import train
    train(run_id)
