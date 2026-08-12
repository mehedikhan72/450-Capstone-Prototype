import os

from celery import Celery

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")

celery_app = Celery("ddos_worker", broker=CELERY_BROKER_URL, include=["app.tasks"])
celery_app.conf.task_default_queue = "ddos_jobs"
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_acks_late = True
celery_app.conf.broker_connection_retry_on_startup = True
