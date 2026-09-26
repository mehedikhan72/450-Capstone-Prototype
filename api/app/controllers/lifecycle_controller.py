from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.controllers.feedback_controller import require_feedback_key
from app.services import lifecycle_service


router = APIRouter(prefix="/learning", tags=["learning"], dependencies=[Depends(require_feedback_key)])


class Actor(BaseModel):
    actor: str = Field(min_length=1, max_length=200)


@router.get("/status")
def learning_status():
    return lifecycle_service.status()


@router.post("/train", status_code=202)
def start_training():
    try: return lifecycle_service.start_training()
    except ValueError as error: raise HTTPException(400, str(error)) from error


@router.post("/models/{model_id}/promote")
def promote(model_id: str, body: Actor):
    try: return lifecycle_service.promote(model_id, body.actor)
    except ValueError as error: raise HTTPException(400, str(error)) from error


@router.post("/rollback")
def rollback(body: Actor):
    try: return lifecycle_service.rollback(body.actor)
    except ValueError as error: raise HTTPException(400, str(error)) from error
