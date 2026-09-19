from fastapi import APIRouter

from app.services import log_service

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("/recent")
def recent_logs():
    runs = log_service.get_recent(limit=5)
    return {"count": len(runs), "runs": runs}
