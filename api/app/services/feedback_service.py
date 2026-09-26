import json
from collections import Counter

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models.feedback import Annotation, Prediction


VALID_LABELS = {"Benign", "Malicious"}


def persist_predictions(job_id: str, model_version: str, frame) -> int:
    """Persist one reviewable observation per output row, once per job."""
    records = json.loads(frame.to_json(orient="records"))
    seen = Counter()
    session = SessionLocal()
    try:
        if session.scalar(select(Prediction.id).where(Prediction.job_id == job_id).limit(1)):
            return 0

        for row_number, record in enumerate(records):
            flow_id = str(record.get("flow_id") or "").strip() or None
            base_key = flow_id or f"row-{row_number}"
            seen[base_key] += 1
            source_key = base_key if seen[base_key] == 1 else f"{base_key}#{seen[base_key]}"

            outputs = {}
            if "pred_binary" in record:
                mode = str(record.get("mode") or "unknown")
                outputs[mode] = {
                    "binary": record.get("pred_binary"),
                    "label": record.get("pred_3label"),
                }
            else:
                for mode in ("HAM", "FDM", "DFDM"):
                    key = f"pred_binary_{mode}"
                    if key in record:
                        outputs[mode] = {
                            "binary": record.get(key),
                            "label": record.get(f"pred_3label_{mode}"),
                        }

            row_data = {
                key: value for key, value in record.items()
                if key != "model_version"
                and not key.startswith(("pred_binary", "pred_3label", "mode"))
            }
            session.add(Prediction(
                job_id=job_id,
                source_row_key=source_key,
                flow_id=flow_id,
                model_version=model_version,
                original_label=record.get("Label"),
                outputs=outputs,
                row_data=row_data,
            ))
        session.commit()
        return len(records)
    finally:
        session.close()


def list_predictions(limit: int = 100, offset: int = 0, review_status: str = "all") -> dict:
    session = SessionLocal()
    try:
        query = select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id)
        has_annotation = select(Annotation.id).where(
            Annotation.prediction_id == Prediction.id
        ).exists()
        if review_status == "reviewed":
            query = query.where(has_annotation)
        elif review_status == "unreviewed":
            query = query.where(~has_annotation)

        total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
        rows = list(session.scalars(query.offset(offset).limit(limit)))
        prediction_ids = [row.id for row in rows]
        latest = {}
        if prediction_ids:
            for annotation in session.scalars(
                select(Annotation)
                .where(Annotation.prediction_id.in_(prediction_ids))
                .order_by(Annotation.created_at, Annotation.id)
            ):
                latest[annotation.prediction_id] = annotation
        return {
            "total": total,
            "predictions": [_serialize_prediction(row, latest.get(row.id)) for row in rows],
        }
    finally:
        session.close()


def annotate(items: list[dict]) -> list[dict]:
    session = SessionLocal()
    try:
        prediction_ids = {item["prediction_id"] for item in items}
        existing = set(session.scalars(
            select(Prediction.id).where(Prediction.id.in_(prediction_ids))
        ))
        missing = sorted(prediction_ids - existing)
        if missing:
            raise ValueError(f"unknown prediction IDs: {missing}")

        created = []
        for item in items:
            if item["label"] not in VALID_LABELS:
                raise ValueError("annotation label must be Benign or Malicious")
            annotation = Annotation(**item)
            session.add(annotation)
            created.append(annotation)
        session.commit()
        return [_serialize_annotation(item) for item in created]
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _serialize_annotation(annotation: Annotation | None):
    if annotation is None:
        return None
    return {
        "id": annotation.id,
        "label": annotation.label,
        "reviewer": annotation.reviewer,
        "reason": annotation.reason,
        "created_at": annotation.created_at.isoformat(),
    }


def _serialize_prediction(prediction: Prediction, annotation: Annotation | None) -> dict:
    return {
        "id": prediction.id,
        "job_id": prediction.job_id,
        "source_row_key": prediction.source_row_key,
        "flow_id": prediction.flow_id,
        "model_version": prediction.model_version,
        "original_label": prediction.original_label,
        "outputs": prediction.outputs,
        "row_data": prediction.row_data,
        "created_at": prediction.created_at.isoformat(),
        "annotation": _serialize_annotation(annotation),
    }
