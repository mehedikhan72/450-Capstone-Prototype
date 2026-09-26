import os
import secrets
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.services import feedback_service


FEEDBACK_API_KEY = os.environ.get("FEEDBACK_API_KEY")


def require_feedback_key(x_feedback_key: str | None = Header(default=None)):
    if FEEDBACK_API_KEY and not (
        x_feedback_key and secrets.compare_digest(x_feedback_key, FEEDBACK_API_KEY)
    ):
        raise HTTPException(401, "invalid feedback API key")


router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    dependencies=[Depends(require_feedback_key)],
)


class AnnotationInput(BaseModel):
    prediction_id: str
    label: Literal["Benign", "Malicious"]
    reviewer: str = Field(min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=2000)


class AnnotationBatch(BaseModel):
    annotations: list[AnnotationInput] = Field(min_length=1, max_length=1000)


@router.get("/predictions")
def predictions(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    review_status: Literal["all", "reviewed", "unreviewed"] = "all",
):
    return feedback_service.list_predictions(limit, offset, review_status)


@router.post("/annotations", status_code=201)
def create_annotations(batch: AnnotationBatch):
    try:
        annotations = feedback_service.annotate(
            [item.model_dump() for item in batch.annotations]
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return {"count": len(annotations), "annotations": annotations}
