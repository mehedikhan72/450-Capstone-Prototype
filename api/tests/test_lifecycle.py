"""Runnable check for candidate promotion and asynchronous fallback rollback."""
import os
import sys
import tempfile


api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, api_dir)
database_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
os.environ["DATABASE_URL"] = f"sqlite:///{database_path}"

from app.db.session import SessionLocal, init_db  # noqa: E402
from app.models.lifecycle import DeploymentState, ModelVersion  # noqa: E402
from app.services import lifecycle_service  # noqa: E402


def main():
    init_db()
    with SessionLocal() as session:
        state = session.get(DeploymentState, 1)
        session.add_all([
            ModelVersion(id="v1", slot="blue", status="current", artifact_path="/v1", artifact_sha256="1"),
            ModelVersion(id="v2", parent_id="v1", slot="green", status="candidate", artifact_path="/v2", artifact_sha256="2"),
        ])
        state.current_model_id = "v1"
        session.commit()

    promoted = lifecycle_service.promote("v2", "test")
    assert promoted["deployment"] == {"active_slot": "green", "current": "v2", "fallback": "v1"}
    with SessionLocal() as session:
        session.add(ModelVersion(id="v3", parent_id="v2", slot="blue", status="candidate",
                                 artifact_path="/v3", artifact_sha256="3"))
        session.commit()

    sent = []
    lifecycle_service.celery_app.send_task = lambda *args, **kwargs: sent.append((args, kwargs))
    assert lifecycle_service.rollback("test") == {"status": "rollback_pending", "model_id": "v1"}
    assert sent[0][1]["queue"] == "inference_blue"
    lifecycle_service.complete_rollback("v1", "test")
    assert lifecycle_service.status()["deployment"] == {
        "active_slot": "blue", "current": "v1", "fallback": None
    }
    with SessionLocal() as session:
        assert session.get(ModelVersion, "v3").status == "rejected"
    print("model lifecycle check passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        os.unlink(database_path)
