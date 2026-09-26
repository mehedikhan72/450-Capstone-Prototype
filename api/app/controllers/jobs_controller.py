from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.services import job_service
from app.services.job_service import InvalidJobRequest

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("")
async def create_job(
    file: UploadFile = File(...),
    flow_rate: Optional[int] = Form(
        None,
        description="MEASURED flows/s. The mode follows from it via the thresholds "
                    "below. Omit to run all three modes.",
    ),
    flow_threshold_high: Optional[int] = Form(
        None, description="flows/s at or above which FDM replaces HAM. Server default if omitted."
    ),
    flow_threshold_extreme: Optional[int] = Form(
        None, description="flows/s at or above which DFDM replaces FDM. Server default if omitted."
    ),
    label_col: Optional[str] = Form(
        "Label", description="Ground-truth column name, or empty if CSV is unlabeled."
    ),
    benign_label: Optional[str] = Form("Benign"),
):
    try:
        return job_service.create_job(
            file, flow_rate, label_col, benign_label,
            flow_threshold_high, flow_threshold_extreme,
        )
    except InvalidJobRequest as e:
        raise HTTPException(400, str(e))


@router.get("/{job_id}/result")
def get_result(job_id: str):
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    if job.status in ("pending", "running"):
        return JSONResponse(status_code=202, content={
            "job_id": job_id,
            "status": job.status,
            "model_version": job.model_version,
        })

    if job.status == "failed":
        return JSONResponse(
            status_code=500,
            content={
                "job_id": job_id,
                "status": "failed",
                "model_version": job.model_version,
                "error": job.error_message,
            },
        )

    return FileResponse(
        job.output_path, media_type="text/csv", filename=f"predictions_{job_id}.csv"
    )
