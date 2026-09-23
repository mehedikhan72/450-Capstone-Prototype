# ═════════════════════════════════════════════════════════════════════════════
# VMFCVD Inference — User Configuration
# Edit this file before running inference.py
# ═════════════════════════════════════════════════════════════════════════════

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))

# ── Checkpoint directory ──────────────────────────────────────────────────────
# Folder containing the single TRIDENT bundle, trident_v13.joblib, produced by
# Notebooks/14-trident-final.ipynb into ./trident_v13_out/.
# May also be a direct path to the .joblib file.
# Exactly one *.joblib must be present, or loading refuses.
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
# FLOW_RATE is a MEASUREMENT in flows/s, not a selector. The mode follows from it:
#     rate >= FLOW_THRESHOLD_EXTREME  -> DFDM
#     rate >= FLOW_THRESHOLD_HIGH     -> FDM
#     otherwise                       -> HAM
#     None                            -> run all three and emit suffixed columns
FLOW_RATE = None

# ── Mode-switching thresholds ─────────────────────────────────────────────────
# DEFAULTS ONLY. These are the deployment contract the models were trained and
# budgeted against (04c Cell 02: budget = 1e6 / threshold). A request may override
# them per job — the ns-3 control frontend does, because a packet-level simulation
# cannot reach carrier flow rates (Ch.1 assumption A8) and the Azure VM cannot be
# reconfigured mid-demonstration.
#
# Overriding is SAFE for the latency argument: a lower threshold yields a LARGER
# budget (1e6/rate), so every guard passes more easily. It is not safe for the
# reported contract, which is why every job records the thresholds it ran under.
FLOW_THRESHOLD_HIGH    = int(_os.environ.get('FLOW_THRESHOLD_HIGH', 1000))
FLOW_THRESHOLD_EXTREME = int(_os.environ.get('FLOW_THRESHOLD_EXTREME', 5000))

# ── Single-row values for Mode B ─────────────────────────────────────────────
# The 12 columns TRIDENT v13 actually reads: FDM's 2 plus HAM's 11, sharing
# `Packet Length Min`. Names must match exactly; order does not matter, because
# the engine selects by name. `Init * Win Bytes = -1` means "no TCP window" and
# is CORRECT for a UDP flow — do not replace it with 0.
SINGLE_ROW = {
    'Packet Length Min'       : 0.0,
    'ACK Flag Count'          : 0.0,
    'Fwd Packets/s'           : 120000.0,
    'Down/Up Ratio'           : 0.0,
    'URG Flag Count'          : 0.0,
    'Fwd Packets Length Total': 6840.0,
    'Init Fwd Win Bytes'      : -1.0,
    'Init Bwd Win Bytes'      : -1.0,
    'Total Fwd Packets'       : 5.0,
    'Fwd Packet Length Max'   : 1460.0,
    'Bwd Packet Length Max'   : 0.0,
    'Avg Packet Size'         : 84.5,
}

# ── Resource monitor settings ─────────────────────────────────────────────────
N_REPEATS  = 5     # inference repeats for timing (more = more accurate)
BATCH_SIZE = 4096  # only used in Mode D
