# Porting the inference engine from V6 to TRIDENT v13

*23 September 2026. Read this before touching `detection-engine/`.*

The engine was built against **V6** — three `vmfcvd_*_step*.pkl` checkpoints and V6's
object graph. The ML side has since moved to **TRIDENT v13** (notebook build `v13.1`),
which ships **one** `joblib` bundle. This note records what changed, why, and what is
now verified.

**Nothing in the FastAPI/Celery/Postgres structure changed.** No new services, no new
endpoints, no change to the job lifecycle or the result CSV.

---

## 1 · Why the port was needed at all

`load_artifacts()` looked for `vmfcvd_*_step10_vmfcvd.pkl` and would not find it. The
v13 bundle is a single file holding `{pre, pre_deploy, fdm, ham, dfdm, schema}`, and
its objects are a different set of classes. Nothing about the old loader could be
adjusted — it had to be replaced.

## 2 · What changed, file by file

| File | Change | Why |
| :-- | :-- | :-- |
| `detection-engine/models.py` | **Replaced** with v13's five classes: `LinRegImputer`, `Pre`, `Member`, `Trident`, `DFDM` | A pickle stores *state*, not behaviour. These definitions are what give the bundle behaviour, so they are copied verbatim from the notebook. V6's serving helpers (`DetailedResourceMonitor`, `compute_metrics*`, `print_system_info`) are kept unchanged. |
| `detection-engine/models_v6_legacy.py` | **New** — the old file, untouched | So the V6 object graph is not lost, and V6 checkpoints stay loadable if anyone needs them. |
| `detection-engine/inference.py` | `_find_prefix` → `_find_bundle`; `load_artifacts` rewritten; `preprocess` steps 4–7 → one `pre_deploy.transform()`; `_predict_one_mode` rewritten | One file instead of three; v13 imputes by regression and transforms with Yeo-Johnson, neither of which the old manual scaling could express. |
| `app/services/inference_service.py` | Five class names in `_bind_main_module_classes()`; `member_report` added to the log payload | The bundle was pickled from a notebook, so its classes resolve as `__main__.Trident` etc. and must be re-bound under Celery. |
| `detection-engine/config.py` | `CKPT_DIR` docs; `SINGLE_ROW` → the real 12 columns; **two threshold settings** | See §4. |
| `app/models/job.py`, `app/db/session.py`, `app/services/job_service.py`, `app/controllers/jobs_controller.py` | Per-request mode thresholds | See §4. |
| `requirements.txt` | `+lightgbm==4.6.0`, `scipy` → `1.16.3`, `+shap==0.50.0` | See §3. |
| `Dockerfile` | `+libgomp1` | See §3. |
| `tests/` | **New** — synthetic-bundle smoke test | Lets you exercise the whole engine without the 4.4 MB artefact, which is gitignored. |

### Three things that are easy to get wrong

**`HEADLINE_PATH` and `DEFAULT_PATH` in `models.py` are part of the model, not config.**
`Trident.score()` and `predict_deploy()` read them as module globals. Delete them and
prediction raises. Change their values and the service runs happily while answering
differently from the reported results — HAM would go through the stacker (2× slower,
identical output) and FDM through the vote (one fewer attack caught per test split).

**v13 emits label *strings*.** `predict_deploy()` returns `'Benign'` / `'Warning'` /
`'Malicious'` directly, so the old `{0: …, 1: …, 2: …}` mapping was **deleted**, not
adjusted. Keeping it would have mislabelled every row.

**The three predict methods have different arities.** `predict_deploy()` returns 2
values, `Trident.predict3()` returns 7, `DFDM.predict3()` returns 5. The engine serves
from `predict_deploy()` — the notebook proves it identical to the full path on every
test flow, and for HAM it is ~2× cheaper because its Warning channel needs no
calibrated probabilities.

## 3 · Two environment bugs found by testing in the container

**`libgomp1` is missing from `python:*-slim`.** LightGBM links against the OpenMP
runtime. Without it `import lightgbm` dies with
`OSError: libgomp.so.1: cannot open shared object file`, and because the `Boosting`
member is an `LGBMClassifier`, **the bundle cannot be unpickled at all** — the worker
would have crashed on startup on the Azure VM after a clean build. Fixed in the
`Dockerfile`.

**`lightgbm` was absent from `requirements.txt`.** Same root cause, different layer.
The other pins now match what the artefact records in `schema['env']`; the engine
prints a loud warning at startup if the installed versions differ, because unpickling
estimators across versions usually does not raise — it returns an object that runs and
answers slightly differently.

*Note:* the image is `python:3.11-slim` and the bundle was pickled under 3.12.12. Every
ML pin matches and results are bit-identical, so this is currently benign. Worth
aligning to 3.12 rather than relying on it after a future retrain.

## 4 · Mode thresholds are now per request

**Why.** The mode follows the offered flow rate — `>= 5000 → DFDM`, `>= 1000 → FDM`,
else HAM. ns-3 cannot reach carrier flow rates (Chapter 1, assumption A8), and the
Azure VM cannot be reconfigured mid-demonstration, so the switching thresholds have to
be settable from the control frontend per run.

**What changed.**

- `flow_rate` was restricted to `{0, 1000, 5000}` and rejected anything else with a
  400 — a measured 3,200 flows/s failed even though the engine maps it to FDM
  correctly. **Any non-negative rate is now accepted**; `None` still runs all three modes.
- `flow_threshold_high` and `flow_threshold_extreme` are new optional form fields.
  Missing values fall back to server config (`FLOW_THRESHOLD_HIGH` /
  `FLOW_THRESHOLD_EXTREME`, default 1000 / 5000).
- Both are **stored on the job row**, so a mode can be reconstructed later. That needed
  two new columns, added by an idempotent `ADD COLUMN IF NOT EXISTS` at startup —
  `create_all()` creates missing *tables* but never alters an existing one, so without
  it a redeploy over a live `pgdata` volume would start cleanly and then fail on every
  insert.
- Invalid rules are rejected at the POST, not inside the job: `extreme <= high`,
  negatives.

**The design point worth keeping.** ns-3 sends the rule's *parameters*; the engine
still makes the decision, and every run records the rule it used. The alternative —
ns-3 sending the mode — would mean ns-3 chooses and the "adaptive switching" claim
weakens to a restatement.

**Safe for the latency argument.** A budget is `1e6 ÷ threshold`, so a *lower*
threshold gives a *larger* budget and every guard passes more easily. It is **not**
safe for the reported contract, which is why each run logs:

```
[mode] flow_rate=35 flows/s  thresholds: HAM < 20 <= FDM < 60 <= DFDM  ->  FDM
[mode] *** these are NOT the trained thresholds (1,000/5,000). Latency budgets are
       derived from the trained pair; this run is a scaled demonstration and must be
       reported as one. ***
```

`/logs/recent` carries `thresholds`, `mode_resolved`, and — only when overridden —
`thresholds_trained` and `thresholds_overridden`.

## 5 · API contract

**Request — `POST /jobs`**

| Field | Before | Now |
| :-- | :-- | :-- |
| `file` | CSV, 6 required columns | CSV, **12** required columns, different names |
| `flow_rate` | selector, only `0`/`1000`/`5000` | **measurement** in flows/s, any non-negative int |
| `flow_threshold_high` | — | new, optional |
| `flow_threshold_extreme` | — | new, optional |
| `label_col`, `benign_label` | unchanged | unchanged |

The 12 columns — names exact, **order irrelevant, extra columns ignored**:

```
Total Fwd Packets · Fwd Packets Length Total · Fwd Packet Length Max
Bwd Packet Length Max · Packet Length Min · Avg Packet Size · Fwd Packets/s
Down/Up Ratio · ACK Flag Count · URG Flag Count
Init Fwd Win Bytes · Init Bwd Win Bytes
```

`Init * Win Bytes = -1` means "no TCP window" and is **correct** for a UDP flow. Do not
clean it to 0 — the model was trained on that distinction.

**Response — `POST /jobs`** gains `flow_rate` and `thresholds`; otherwise unchanged.

**`GET /jobs/{id}/result` is completely unchanged** — same status codes, same CSV:
input columns passed through plus `pred_binary`, `pred_3label`, `mode` (suffixed per
mode when `flow_rate` is omitted). `pred_3label` is still the same three strings, and
DFDM still never emits `Warning`. **No change is needed in the ns-3 result parser.**

**`GET /logs/recent`** is additive: the threshold fields above, plus a `member_report`
per base model carrying its Shapley weight, per-sample cost and — **only when the
upload carried a `Label`** — its own MCC, recall, precision and confusion counts. With
no label those keys are *absent*, not zero; do not write a frontend that assumes them.

## 6 · Verification

Run inside the container, on the real bundle and the notebook's own held-out split
(`split_test.parquet` → CSV, 25,462 rows), with all three modes:

| Mode | TP | FN | FP | TN | MCC | Warnings | |
| :-- | --: | --: | --: | --: | --: | --: | :-- |
| FDM | 11,174 | 104 | 4 | 14,180 | 0.991428 | 228 | match |
| **HAM** | **11,270** | **8** | **1** | **14,183** | **0.999284** | **34** | **match** |
| DFDM | 11,174 | 104 | 4 | 14,180 | 0.991428 | 0 | match |

**Exact replication of the notebook, to the last flow.** Every operating point also
matched on load: both Shapley vectors, `MaxVoteIndex` 0.2975 / 0.4775,
`dissent >= 0.116762`, `spread >= 0.0288838`, and both headline paths.

Also verified: the threshold rule at default and demo settings with inclusive `>=`
boundaries; partial override; invalid rules rejected; 12 / reordered / extra-column
inputs all accepted; a missing column named in the error; `assert_schema`'s negative
control; and that an unlabelled upload emits no metric keys.

**Not yet exercised:** the HTTP path end to end — `POST /jobs` through Celery and
Postgres. The engine was tested exactly as the worker calls it, not through the queue.

## 7 · Running it

```bash
mkdir -p api/detection-engine/weights
cp <somewhere>/trident_v13.joblib api/detection-engine/weights/
cd api && docker compose up --build
```

`weights/` is gitignored, so **the model is not in git** — keep a copy elsewhere.
Exactly one `*.joblib` may be present; the loader refuses if it finds two.

To smoke-test without the real artefact, see `tests/README.md`. Note the generator
writes to the same filename as the real bundle — delete it before copying the real one
in.
