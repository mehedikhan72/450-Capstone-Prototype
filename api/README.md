# DDoS Detection API

FastAPI + Celery/RabbitMQ job queue around the VMFCVD detection model. Upload a
flow CSV, poll for results.

## Run

```bash
cd api
docker compose up -d --build
```

Starts Postgres, RabbitMQ, the API (`:8000`), and a Celery worker. The worker
loads the model once on startup (`detection-engine/weights/*.pkl` must be
present — copy them in if missing, they're gitignored).

## Endpoints

### `POST /jobs`

Submit a CSV for detection. Multipart form:

| field          | required | default   | notes                                              |
|----------------|----------|-----------|-----------------------------------------------------|
| `file`         | yes      | —         | the flow CSV                                       |
| `flow_rate`    | no       | all modes | `0`=HAM, `1000`=FDM, `5000`=DFDM. Omit the field entirely for all three (don't send an empty string). |
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

### `GET /health`

Liveness check, returns `{"status": "ok"}`.
