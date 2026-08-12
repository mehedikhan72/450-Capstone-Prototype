# ═════════════════════════════════════════════════════════════════════════════
# VMFCVD Inference — User Configuration
# Edit this file before running inference.py
# ═════════════════════════════════════════════════════════════════════════════

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

# ── Checkpoint directory ──────────────────────────────────────────────────────
# Folder containing vmfcvd_*_step*.pkl files from the training run.
# On Kaggle: typically '/kaggle/working' or '/kaggle/input/<dataset-name>'
CKPT_DIR = _os.path.join(_HERE, 'weights')

# ── Input / output CSV paths ──────────────────────────────────────────────────
CSV_PATH = _os.path.join(_HERE, 'input', 'flows.csv')
OUT_PATH = _os.path.join(_HERE, 'output', 'predictions.csv')

# ── Label configuration ───────────────────────────────────────────────────────
# Set LABEL_COL to None if your CSV has no ground-truth column (predictions only).
# If labels are strings ('Benign', 'DrDoS_DNS', …) set BENIGN_LABEL accordingly.
# If labels are already binary integers (0/1), BENIGN_LABEL is ignored.
LABEL_COL    = 'Label'   # or None
BENIGN_LABEL = 'Benign'

# ── Mode switching ────────────────────────────────────────────────────────────
# None  → all three modes run (same as Mode C)
# 0     → HAM  (stable / normal traffic)
# 1000  → FDM  (high volume / DDoS suspected)
# 5000  → DFDM (extreme DDoS, emergency)
FLOW_RATE = None

# ── Single-row values for Mode B ─────────────────────────────────────────────
# Fill these after running once to see which features are required.
SINGLE_ROW = {
    'Init Fwd Win Bytes'      : 65535.0,
    'Avg Packet Size'         : 84.5,
    'Fwd Packet Length Max'   : 1460.0,
    'Fwd Packets Length Total': 6840.0,
    'Flow IAT Mean'           : 1200.0,
    'Init Bwd Win Bytes'      : 65535.0,
}

# ── Resource monitor settings ─────────────────────────────────────────────────
N_REPEATS  = 5     # inference repeats for timing (more = more accurate)
BATCH_SIZE = 4096  # only used in Mode D
