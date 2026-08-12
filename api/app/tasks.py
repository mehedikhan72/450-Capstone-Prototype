from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.services import inference_service


@worker_process_init.connect
def _load_model(**kwargs):
    inference_service.load_model()


@celery_app.task(name="run_prediction_job")
def run_prediction_job(job_id: str):
    inference_service.run_prediction(job_id)
