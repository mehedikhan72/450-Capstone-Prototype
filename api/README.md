# DDoS Detection API

FastAPI + Celery/RabbitMQ job queue around the TRIDENT detection model. Upload a
flow CSV, poll for results.

## Run

```bash
cd api
docker compose up -d --build
```

Starts Postgres, RabbitMQ, the API (loopback-only `127.0.0.1:8000`), blue/green
inference workers, and a separate training worker. The inference workers load the
bootstrap model once on startup (`detection-engine/weights/*.joblib` must be
present — copy it in if missing; model artifacts are gitignored).
The training worker reads the repository-owned `data/DDoS_Dataset.zip` as the
immutable base dataset and records its SHA-256 in every snapshot.

## Single public port

Nothing in this stack is published publicly. The dashboard backend (the
`capstone` repo) is the only public port (default `:8765`): it serves the UI and
proxies `/api/jobs`, `/api/feedback/*`, and `/api/learning/*` to this API over
`127.0.0.1:8000` (`LEARNING_API_BASE`). The ns-3 controller targets the same
port (`--api-base http://HOST:8765`). On a single-port host (Azure App Service
etc.), expose only the dashboard port.

## Endpoints

### `POST /jobs`

Submit a CSV for detection. Multipart form:

| field          | required | default   | notes                                              |
|----------------|----------|-----------|-----------------------------------------------------|
| `file`         | yes      | —         | the flow CSV                                       |
| `flow_rate`    | no       | all modes | measured flows/s; the configured thresholds select HAM/FDM/DFDM. Omit the field entirely for all three. |
| `label_col`    | no       | `Label`   | ground-truth column name, if present                |
| `benign_label` | no       | `Benign`  | value in `label_col` meaning "not an attack"        |

```bash
curl -X POST http://localhost:8000/jobs -F "file=@input/flows.csv"
# -> {"job_id": "...", "status": "pending"}
```

### `GET /jobs/{job_id}/result`

Poll for the result.

- `202` — still `pending`/`running`
- `200` — done, body is `predictions.csv`
- `500` — failed, body has `error`
- `404` — unknown job id

```bash
curl http://localhost:8000/jobs/<job_id>/result -o predictions.csv
```

### `GET /logs/recent`

Returns the last 5 inference runs (most recent first), each with row count,
modes run, per-mode prediction breakdown, binary/3-label metrics (when the
CSV had a label column), run duration, and a `models` breakdown — each base
model in that mode's ensemble (e.g. AdaBoost/Bagging/GradientBoosting/
RandomForest/ExtraTrees for HAM) with its own pre-voting predictions and
accuracy. Appended to on every job, success or failure, by the worker. This
is logging only — the ensemble's actual prediction/predictions.csv is
unaffected.

```bash
curl http://localhost:8000/logs/recent
```

### `GET /feedback/predictions`

Returns prediction rows with their model version and latest human annotation.
Supports `limit`, `offset`, and `review_status=all|reviewed|unreviewed`.
Set `FEEDBACK_API_KEY` in the API environment to require the
`X-Feedback-Key` header on feedback endpoints.

### `POST /feedback/annotations`

Appends immutable human labels. Ground truth is strictly `Benign` or
`Malicious`; `Warning` is a model output, not a review label.

```json
{
  "annotations": [{
    "prediction_id": "...",
    "label": "Malicious",
    "reviewer": "operator@example.com",
    "reason": "verified attack"
  }]
}
```

### Model learning

- `GET /learning/status` — deployment, model, and recent training-run state.
- `POST /learning/train` — freeze reviewed labels into a dataset snapshot and
  queue a candidate retrain while the active slot keeps serving.
- `POST /learning/models/{id}/promote` with `{"actor":"..."}` — atomically
  route new jobs to a validated candidate and retain the old current as fallback.
- `POST /learning/rollback` with `{"actor":"..."}` — reload and route to the
  fallback without mutating the active model in place.

New reviews after a run's cutoff are reserved for the next run. Candidates must
pass locked-test MCC, recall, false-positive, and latency gates before promotion.

### `GET /health`

Liveness check, returns `{"status": "ok"}`.
