import json
import os

import numpy as np

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
LOG_PATH = os.path.join(OUTPUT_DIR, "inference_log.jsonl")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_entry(entry: dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=_json_default) + "\n")


def get_recent(limit: int = 5) -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        lines = [line for line in f if line.strip()]
    recent = lines[-limit:]
    entries = [json.loads(line) for line in recent]
    entries.reverse()  # most recent first
    return entries
