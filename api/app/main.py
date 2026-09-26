from fastapi import FastAPI

from app.controllers import feedback_controller, health_controller, jobs_controller, lifecycle_controller, logs_controller
from app.db.session import init_db
from app.services.lifecycle_service import ensure_bootstrap_model

app = FastAPI(title="DDoS Detection API")
app.include_router(health_controller.router)
app.include_router(jobs_controller.router)
app.include_router(logs_controller.router)
app.include_router(feedback_controller.router)
app.include_router(lifecycle_controller.router)


@app.on_event("startup")
def _startup():
    init_db()
    ensure_bootstrap_model()
