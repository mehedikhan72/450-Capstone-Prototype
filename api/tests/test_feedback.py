"""Small runnable check for prediction persistence and immutable corrections."""
import os
import json
import sys
import tempfile


api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, api_dir)
database_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
os.environ["DATABASE_URL"] = f"sqlite:///{database_path}"

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.models.job import Job  # noqa: E402
from app.services import feedback_service  # noqa: E402


def main():
    init_db()
    session = SessionLocal()
    session.add(Job(
        id="job-1", status="done", input_filename="flows.csv", input_path="flows.csv",
        model_version="v13-deadbeef",
    ))
    session.commit()
    session.close()

    records = [
        {
            "flow_id": "flow-1", "src_ip": "10.0.0.1", "Label": "Benign",
            "pred_binary": 1, "pred_3label": "Malicious", "mode": "HAM",
        },
        {
            "flow_id": "flow-2", "src_ip": "10.0.0.2", "Label": "Malicious",
            "pred_binary": 1, "pred_3label": "Warning", "mode": "HAM",
        },
    ]

    class Frame:
        def to_json(self, orient):
            assert orient == "records"
            return json.dumps(records)

    frame = Frame()
    assert feedback_service.persist_predictions("job-1", "v13-deadbeef", frame) == 2
    assert feedback_service.persist_predictions("job-1", "v13-deadbeef", frame) == 0

    pending = feedback_service.list_predictions(review_status="unreviewed")
    assert pending["total"] == 2
    first_id = pending["predictions"][0]["id"]
    feedback_service.annotate([{
        "prediction_id": first_id,
        "label": "Malicious",
        "reviewer": "operator@example.test",
        "reason": "simulation ground truth",
    }])
    feedback_service.annotate([{
        "prediction_id": first_id,
        "label": "Benign",
        "reviewer": "reviewer@example.test",
        "reason": "correction",
    }])

    reviewed = feedback_service.list_predictions(review_status="reviewed")
    assert reviewed["total"] == 1
    assert reviewed["predictions"][0]["annotation"]["label"] == "Benign"
    assert feedback_service.list_predictions(review_status="unreviewed")["total"] == 1
    print("feedback persistence check passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        os.unlink(database_path)
