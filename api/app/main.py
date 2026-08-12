from fastapi import FastAPI

from app.controllers import health_controller, jobs_controller
from app.db.session import init_db

app = FastAPI(title="DDoS Detection API")
app.include_router(health_controller.router)
app.include_router(jobs_controller.router)


@app.on_event("startup")
def _startup():
    init_db()
