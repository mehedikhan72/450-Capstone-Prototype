#!/usr/bin/env python3
"""
TRIDENT Inference
=================
Compatible with: TRIDENT v13 (build v13.1) — one joblib bundle holding
{pre, pre_deploy, fdm, ham, dfdm, schema}. FDM/DFDM 4 members over 2 features,
HAM 5 members over 11; the live path needs 12 columns (the union).

Usage
-----
    python inference.py              # runs Mode A (CSV) with settings from config.py
    python inference.py --mode a     # Mode A: CSV inference
    python inference.py --mode b     # Mode B: single-row inference
    python inference.py --mode c     # Mode C: all-modes comparison (HAM/FDM/DFDM)
    python inference.py --mode d     # Mode D: batch predictor (large files only)

Edit config.py to set paths, FLOW_RATE, SINGLE_ROW, etc.
"""

import argparse
import os
import glob
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)

# Project modules
from models import (
    LinRegImputer, Pre, Member, Trident, DFDM, DetailedResourceMonitor,
    compute_metrics, compute_metrics_3label, print_system_info,
    USE_EARLY_EXIT, HEADLINE_PATH, DEFAULT_PATH,
)
import config as cfg

# ── Global session state ──────────────────────────────────────────────────────
INFER_MONITOR: DetailedResourceMonitor = None
PRE                                     = None   # 12-column deployment preprocessor
FDM_ENS:  Trident                       = None
HAM_ENS:  Trident                       = None
DFDM_ENS: DFDM                          = None
SCHEMA                                  = None
FDM_FEATURES                            = None
HAM_FEATURES                            = None
ALL_FEATURES                            = None   # the 12-column union the CSV must carry


# ═════════════════════════════════════════════════════════════════════════════
# Artifact loading
# ═════════════════════════════════════════════════════════════════════════════

def _find_bundle(ckpt_dir: str) -> str:
    """Locate the single TRIDENT .joblib bundle. One file, not three pickles."""
    if os.path.isfile(ckpt_dir):
        return ckpt_dir
    matches = sorted(glob.glob(os.path.join(ckpt_dir, '*.joblib')))
    if not matches:
        raise FileNotFoundError(
            f'No .joblib bundle in "{ckpt_dir}".\n'
            f'Copy trident_v13.joblib (from the notebook\'s trident_v13_out/) there.\n'
            f'Files found: {os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else "DIR NOT FOUND"}')
    if len(matches) > 1:
        raise RuntimeError(
            f'Multiple bundles found: {[os.path.basename(m) for m in matches]}.\n'
            f'Keep only the one you want in CKPT_DIR.')
    return matches[0]


def load_artifacts(ckpt_dir: str = cfg.CKPT_DIR) -> dict:
    global INFER_MONITOR, PRE, FDM_ENS, HAM_ENS, DFDM_ENS, SCHEMA
    global FDM_FEATURES, HAM_FEATURES, ALL_FEATURES

    import joblib

    INFER_MONITOR = DetailedResourceMonitor(poll_interval=0.05)
    path = _find_bundle(ckpt_dir)
    print(f'[ckpt] Using bundle: {os.path.basename(path)}')

    INFER_MONITOR.start('Load_bundle')
    bundle = joblib.load(path)
    INFER_MONITOR.stop('Load_bundle')

    missing = [k for k in ('fdm', 'ham', 'dfdm', 'schema') if k not in bundle]
    if missing:
        raise KeyError(f'Bundle is missing {missing}; keys present: {sorted(bundle)}')

    schema = bundle['schema']
    fdm, ham, dfdm = bundle['fdm'], bundle['ham'], bundle['dfdm']

    # pre_deploy is the 12-column preprocessor, proven bit-identical to the
    # 59-column `pre` on the notebook's test split. Fall back to `pre` only if an
    # older bundle predates it -- that path then needs all 59 columns.
    pre = bundle.get('pre_deploy')
    if pre is None:
        pre = bundle['pre']
        print('[ckpt] WARNING: bundle has no pre_deploy; falling back to the '
              '59-column preprocessor. The input CSV must then carry all 59 features.')

    fdm_features = list(schema['fdm_features'])
    ham_features = list(schema['ham_features'])
    all_features = list(schema.get('deploy_features') or pre.cols)

    # ---- startup audit trail: what exactly is being served -------------------
    print(f'\n[verify] {schema.get("version", "?")} — {schema.get("build", "no build tag")}')
    env = schema.get('env') or {}
    if env:
        print('[verify] trained with: ' + '  '.join(
            f'{k}={env[k]}' for k in ('python', 'numpy', 'pandas', 'scikit_learn', 'lightgbm')
            if env.get(k)))
        _env_warnings(env)
    for tag, ens in (('FDM', fdm), ('HAM', ham)):
        path_used = HEADLINE_PATH.get(ens.mode, DEFAULT_PATH)
        print(f'[verify] {tag} voter:')
        print(f'  members      : {[m.name for m in ens.members]}  ({len(ens.members)} models)')
        print(f'  shapley      : ' + ', '.join(
            f'{m.name}={w:.4f}' for m, w in zip(ens.members, ens.w)))
        print(f'  MaxVoteIndex : {ens.maxvote:.4f}')
        print(f'  Warning head : dissent>={ens.dissent_min:.6g}  spread>={ens.tau:.6g}'
              f'   (inf = channel off)')
        print(f'  headline path: {path_used}'
              + ('   [deployment fast path applies]'
                 if path_used == 'vote' and not np.isfinite(ens.tau) else ''))
    print(f'[features] FDM  : {fdm_features}')
    print(f'[features] HAM  : {ham_features}')
    print(f'[features] UNION: {all_features}  ← minimum columns your CSV needs '
          f'({len(all_features)})')

    df_load = INFER_MONITOR.summary_df()
    if not df_load.empty:
        print('\n[resource] Loading stage times:')
        print(df_load[['Stage', 'Time (s)', 'Mem Δ (MB)', 'Peak Δ (MB)', 'Peak Mem (MB)']]
              .to_string(index=False))

    PRE, FDM_ENS, HAM_ENS, DFDM_ENS, SCHEMA = pre, fdm, ham, dfdm, schema
    FDM_FEATURES, HAM_FEATURES, ALL_FEATURES = fdm_features, ham_features, all_features

    return dict(pre=pre, fdm=fdm, ham=ham, dfdm=dfdm, schema=schema,
                fdm_features=fdm_features, ham_features=ham_features,
                all_features=all_features)


def _env_warnings(env: dict) -> None:
    """Loud, non-fatal warning when the serving libraries differ from training.

    Unpickling estimators across scikit-learn or LightGBM versions usually does not
    raise -- it returns an object that runs and answers slightly differently. That is
    the failure this check exists to make visible.
    """
    import sklearn
    checks = [('scikit_learn', 'scikit-learn', sklearn.__version__)]
    checks.append(('numpy', 'numpy', np.__version__))
    checks.append(('pandas', 'pandas', pd.__version__))
    try:
        import lightgbm as _lgb
        checks.append(('lightgbm', 'lightgbm', _lgb.__version__))
    except ImportError:
        if env.get('lightgbm'):
            print('[verify] *** lightgbm is NOT INSTALLED but the bundle was trained '
                  f'with {env["lightgbm"]}. The Boosting member cannot unpickle. ***')
    for key, label, running in checks:
        trained = env.get(key)
        if trained and trained != running:
            print(f'[verify] *** VERSION MISMATCH: {label} trained={trained} '
                  f'running={running} — predictions may differ silently ***')


def assert_schema(df: pd.DataFrame, mode: str = 'HAM') -> bool:
    """The notebook's Cell 31 contract, applied to an already-ordered frame.

    Call it AFTER selecting the model's columns in schema order, never on the raw
    upload: it tests order as well as names, and a natural CSV order would be
    rejected for no good reason.
    """
    want = list(SCHEMA['fdm_features'] if mode == 'FDM' else SCHEMA['ham_features'])
    got = [c for c in df.columns if c in want]
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise ValueError(f'{mode}: missing columns {missing}')
    if got != want:
        raise ValueError(f'{mode}: column ORDER mismatch\n  expected {want}\n  got {got}')
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(df_raw: pd.DataFrame,
               label_col=cfg.LABEL_COL,
               benign_label=cfg.BENIGN_LABEL):
    """Apply the same preprocessing as the training pipeline. Returns (X, y_or_None)."""
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    y = None
    if label_col and label_col in df.columns:
        raw = df[label_col].copy()
        if raw.dtype == object or str(raw.dtype) == 'category':
            y = (raw != benign_label).astype(int).values
            print(f'[label] Binarized: "{benign_label}"→0, other→1  '
                  f'| benign={int((y==0).sum()):,}  malicious={int((y==1).sum()):,}')
        else:
            y = raw.astype(int).values
            print(f'[label] Numeric 0/1  '
                  f'| benign={int((y==0).sum()):,}  malicious={int((y==1).sum()):,}')
        df.drop(columns=[label_col], inplace=True)
    elif label_col:
        print(f'[label] Column "{label_col}" not in CSV — predictions only.')

    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f'Missing required columns: {missing}\n'
            f'Your CSV must contain at least these {len(ALL_FEATURES)}: {ALL_FEATURES}')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Overflow sentinels, exactly as the training pipeline treats them (Cell 04):
    # a negative value in these three columns is integer-overflow corruption and
    # becomes missing, to be imputed. None of them is in the deployed 12, so this
    # is a no-op on a 12-column upload and matters only for a 59-column one.
    for c in ('Fwd Header Length', 'Bwd Header Length', 'Fwd Seg Size Min'):
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    # `Init Fwd Win Bytes` / `Init Bwd Win Bytes` = -1 means "no TCP window" and is
    # CORRECT for every UDP flow. It is deliberately NOT cleaned here.

    X = df[ALL_FEATURES].astype(float)

    # One call replaces the old steps 4-7. It applies Algorithm 1's regression
    # imputer and the fitted Yeo-Johnson PowerTransformer together. The previous
    # fillna(0) is gone: v13 imputes by regression on a correlated donor, and a
    # zero fill would silently shift the affected columns. The manual
    # mean_/scale_ scaling is gone too -- PowerTransformer exposes lambdas_.
    X = PRE.transform(X)

    print(f'[preprocess] Ready: {X.shape[0]:,} rows × {X.shape[1]} features')
    return X, y


def single_row_to_df(row_dict: dict) -> pd.DataFrame:
    missing = [f for f in ALL_FEATURES if f not in row_dict]
    if missing:
        raise ValueError(f'SINGLE_ROW missing columns: {missing}\n'
                         f'Required: {ALL_FEATURES}')
    return pd.DataFrame([row_dict])[ALL_FEATURES]


def resolve_thresholds(high=None, extreme=None) -> dict:
    """The switching rule in force for one job: caller's values, else config defaults."""
    h = int(cfg.FLOW_THRESHOLD_HIGH    if high    is None else high)
    e = int(cfg.FLOW_THRESHOLD_EXTREME if extreme is None else extreme)
    if h < 0 or e < 0:
        raise ValueError(f'thresholds must be non-negative (got high={h}, extreme={e})')
    if e <= h:
        raise ValueError(f'flow_threshold_extreme ({e}) must exceed '
                         f'flow_threshold_high ({h})')
    return {'high': h, 'extreme': e}


def _flow_to_mode(flow_rate, thresholds=None):
    """Map a MEASURED flow rate to a mode. The caller supplies the rate and may
    supply the rule; the decision is made here either way, and the rule in force is
    recorded on the job so a run can be reconstructed."""
    if flow_rate is None:
        return None
    t = thresholds or resolve_thresholds()
    if flow_rate >= t['extreme']:
        return 'DFDM'
    if flow_rate >= t['high']:
        return 'FDM'
    return 'HAM' 


_LABEL_CODE = {'Benign': 0, 'Warning': 1, 'Malicious': 2}


def _to_code(labels):
    """v13 emits label STRINGS; compute_metrics_3label still speaks 0/1/2."""
    return pd.Series(labels).map(_LABEL_CODE).to_numpy()


def _mode_ensemble(mode_name: str):
    return {'FDM': FDM_ENS, 'HAM': HAM_ENS, 'DFDM': DFDM_ENS}[mode_name]


def _predict_one_mode(X, mode_name: str, monitor=None):
    """Binary + 3-label prediction for one mode.

    Returns (p_bin, labels) where `labels` are the STRINGS 'Benign'/'Warning'/
    'Malicious'. v13 emits them directly, so the engine's old 0/1/2 -> text map is
    gone rather than adjusted; keeping it would have mislabelled every row.

    FDM and HAM are served through predict_deploy(), the path the notebook's
    latency figures measure and that Cell 21 proves identical to the full path on
    every test flow. For HAM that is ~2x cheaper, because its Warning channel
    needs no calibrated probabilities.

    Members slice their own columns by name, so the whole transformed frame is
    passed; no per-mode column selection is needed.
    """
    if mode_name == 'DFDM':
        if monitor:
            monitor.record_inference('DFDM early-exit',
                                     lambda Z: DFDM_ENS.predict3(Z)[1], X, cfg.N_REPEATS)
        labels, p_bin, _warn, _stage, stats = DFDM_ENS.predict3(X)
        if stats.get('late_catches'):
            print(f'[DFDM] late catches (diagnostic only, not labelled Warning): '
                  f'{stats["late_catches"]:,}')
        return np.asarray(p_bin), np.asarray(labels)

    ens = _mode_ensemble(mode_name)
    if monitor:
        monitor.record_inference(f'{mode_name}  deploy',
                                 lambda Z: ens.predict_deploy(Z)[1], X, cfg.N_REPEATS)
    labels, p_bin = ens.predict_deploy(X)
    return np.asarray(p_bin), np.asarray(labels)


# ═════════════════════════════════════════════════════════════════════════════
# Mode A — CSV inference
# ═════════════════════════════════════════════════════════════════════════════

def run_csv(csv_path=cfg.CSV_PATH, flow_rate=cfg.FLOW_RATE,
            label_col=cfg.LABEL_COL, benign_label=cfg.BENIGN_LABEL,
            save_output=True, track_resources=True,
            flow_threshold_high=None, flow_threshold_extreme=None):
    """
    Mode A: full-file CSV inference.
    If flow_rate is None, all three modes run (same behaviour as Mode C).
    Binary + 3-label metrics are printed when LABEL_COL is present.
    """
    sep = '='*65
    print(sep); print(f'  MODE A — CSV inference: {csv_path}'); print(sep)

    df_raw = pd.read_csv(csv_path)
    print(f'[csv] {len(df_raw):,} rows  {df_raw.shape[1]} columns')

    INFER_MONITOR.start('Preprocess')
    X, y = preprocess(df_raw, label_col, benign_label)
    INFER_MONITOR.stop('Preprocess')

    thresholds = resolve_thresholds(flow_threshold_high, flow_threshold_extreme)
    mode    = _flow_to_mode(flow_rate, thresholds)
    modes   = [mode] if mode else ['HAM', 'FDM', 'DFDM']
    monitor = INFER_MONITOR if track_resources else None
    all_res = {}

    trained = (SCHEMA or {}).get('flow_thresholds') or {}
    print(f'[mode] flow_rate={flow_rate} flows/s  thresholds: HAM < {thresholds["high"]:,} '
          f'<= FDM < {thresholds["extreme"]:,} <= DFDM  ->  '
          f'{mode or "ALL THREE MODES"}')
    if trained and (trained.get('high'), trained.get('extreme')) != (thresholds['high'],
                                                                    thresholds['extreme']):
        print(f'[mode] *** these are NOT the trained thresholds '
              f'({trained.get("high"):,}/{trained.get("extreme"):,}). Latency budgets are '
              f'derived from the trained pair; this run is a scaled demonstration and must '
              f'be reported as one. ***')

    for m in modes:
        print(f'\n{sep}\n  Predictions — {m}\n{sep}')
        t0 = time.perf_counter()
        p_bin, p_lab = _predict_one_mode(X, m, monitor=monitor)
        us = (time.perf_counter() - t0) / len(X) * 1e6
        n_mal = int((p_bin == 1).sum())
        print(f'[{m}] Benign={len(p_bin)-n_mal:,}  Malicious={n_mal:,}  '
              f'({n_mal/len(p_bin)*100:.2f}% attack)  {us:.2f} µs/sample (wall)')
        n_w = int((p_lab == 'Warning').sum())
        print(f'[{m} 3L] Benign={int((p_lab=="Benign").sum())}  Warning={n_w}  '
              f'Malicious={int((p_lab=="Malicious").sum())}  '
              f'(warn_rate={n_w/len(p_lab)*100:.2f}%)'
              + ('   [DFDM is a two-label mode by design]' if m == 'DFDM' else ''))
        res = {'predictions_binary': p_bin, 'predictions_3label': p_lab}
        if y is not None:
            print('\nBinary metrics:')
            res['metrics_binary'] = compute_metrics(y, p_bin, mode_name=m)
            print('3-label metrics:')
            res['metrics_3label'] = compute_metrics_3label(y, _to_code(p_lab), mode_name=m)

        # Per-base-model breakdown (pre-voting) — logging/study only, does not
        # affect predictions_binary/predictions_3label or the saved CSV.
        # v13 ships member_report(), which returns predictions always and metrics
        # ONLY when ground truth is supplied, so an unlabelled upload can never
        # produce a metric scored against nothing.
        ens = _mode_ensemble(m)
        res['model_predictions'] = ens.individual_predictions(X)
        res['model_report'] = ens.member_report(X, y)
        if y is not None:
            res['model_metrics'] = {
                name: compute_metrics(y, preds, mode_name=f'{m}/{name}', verbose=False)
                for name, preds in res['model_predictions'].items()
            }
        all_res[m] = res

    df_out = df_raw.copy()
    for m, r in all_res.items():
        sfx = '' if len(all_res) == 1 else f'_{m}'
        df_out[f'pred_binary{sfx}']  = r['predictions_binary']
        df_out[f'pred_3label{sfx}']  = r['predictions_3label']
        df_out[f'mode{sfx}']         = m

    if save_output:
        # The Dockerfile creates this directory; running inference.py directly on a
        # host does not, so create it here rather than failing after the work is done.
        os.makedirs(os.path.dirname(cfg.OUT_PATH) or '.', exist_ok=True)
        df_out.to_csv(cfg.OUT_PATH, index=False)
        print(f'\n[save] {cfg.OUT_PATH}')

    return all_res, df_out


# ═════════════════════════════════════════════════════════════════════════════
# Mode B — Single-row inference
# ═════════════════════════════════════════════════════════════════════════════

def run_single_row(row_dict=cfg.SINGLE_ROW, flow_rate=cfg.FLOW_RATE, true_label=None):
    """
    Mode B: single-row inference.
    No metrics unless true_label is provided (0 or 1).
    All three modes run when flow_rate is None.
    """
    sep = '='*65
    print(sep); print('  MODE B — Single row inference'); print(sep)

    df_single = single_row_to_df(row_dict)
    X, _ = preprocess(df_single, label_col=None)

    mode    = _flow_to_mode(flow_rate)
    modes   = [mode] if mode else ['HAM', 'FDM', 'DFDM']
    label_b = {0: 'Benign', 1: 'Malicious'}

    for m in modes:
        p_bin, p_lab = _predict_one_mode(X, m, monitor=None)
        pred_b = label_b[int(p_bin[0])]
        pred_3 = str(p_lab[0])
        correct = ''
        if true_label is not None:
            exp = label_b[int(true_label)]
            correct = ('  ✓ correct' if int(p_bin[0]) == int(true_label)
                       else f'  ✗ wrong (expected {exp})')
        print(f'\n  [{m}]')
        print(f'    Binary  : {pred_b}{correct}')
        print(f'    3-label : {pred_3}', end='')
        if pred_3 == 'Warning':
            print('  ← the members disagreed on this flow', end='')
        print()
        if m == 'DFDM':
            print('    DFDM is a two-label mode by design: above '
                  '5,000 flows/s there is no analyst to escalate to')

    print(f'\n  Feature values used:')
    for col in ALL_FEATURES:
        print(f'    {col:35s}: {row_dict.get(col, "MISSING")}')


# ═════════════════════════════════════════════════════════════════════════════
# Mode C — All-modes comparison
# ═════════════════════════════════════════════════════════════════════════════

def run_all_modes(csv_path=cfg.CSV_PATH, label_col=cfg.LABEL_COL,
                  benign_label=cfg.BENIGN_LABEL, track_resources=True):
    """
    Mode C: forces HAM + FDM + DFDM regardless of FLOW_RATE.
    Prints a summary comparison table.
    """
    sep = '='*65
    print(sep); print('  MODE C — All modes comparison (HAM / FDM / DFDM)'); print(sep)

    df_raw = pd.read_csv(csv_path)
    print(f'[csv] {len(df_raw):,} rows')
    X, y    = preprocess(df_raw, label_col, benign_label)
    monitor = INFER_MONITOR if track_resources else None

    summary = []
    for m in ['HAM', 'FDM', 'DFDM']:
        t0 = time.perf_counter()
        p_bin, p_lab = _predict_one_mode(X, m, monitor=monitor)
        us  = (time.perf_counter() - t0) / len(X) * 1e6
        row = {'Mode': m, 'Speed_us(wall)': round(us, 2),
               'Malicious%': round(p_bin.mean() * 100, 2)}
        if y is not None:
            mb = compute_metrics(y, p_bin, mode_name=m)
            row.update({k: round(v, 6) for k, v in mb.items()
                        if k in ('Accuracy', 'Precision', 'Sensitivity', 'F1')})
            m3 = compute_metrics_3label(y, _to_code(p_lab), mode_name=m)
            row['ConfAcc']       = round(m3['ConfidentAccuracy'], 6)
            row['WarnRate%']     = round(m3['WarningRate'] * 100, 3)
            row['AttacksMissed'] = m3['AttacksLetThrough']
        summary.append(row)

    print('\n' + '='*65 + '\n  SUMMARY TABLE\n' + '='*65)
    print(pd.DataFrame(summary).set_index('Mode').to_string())
    return pd.DataFrame(summary)


# ═════════════════════════════════════════════════════════════════════════════
# Mode D — Batch predictor (large files only)
# ═════════════════════════════════════════════════════════════════════════════

def run_batch_predictor(csv_path=cfg.CSV_PATH, flow_rate=cfg.FLOW_RATE,
                        batch_size=cfg.BATCH_SIZE, label_col=cfg.LABEL_COL,
                        benign_label=cfg.BENIGN_LABEL,
                        flow_threshold_high=None, flow_threshold_extreme=None):
    """
    Mode D: mini-batch inference for files too large to load all at once.
    Slower than Mode A for normal-sized files — use only to avoid OOM.
    flow_rate must be a specific integer (not None) — one mode only.
    """
    if flow_rate is None:
        raise ValueError('Batch mode needs a specific flow_rate (one mode only).')
    mode = _flow_to_mode(flow_rate,
                         resolve_thresholds(flow_threshold_high, flow_threshold_extreme))
    print(f'  MODE D — Batch predictor  mode={mode}  batch_size={batch_size}')
    print('  NOTE: Use Mode A for normal files — it is faster.')

    reader              = pd.read_csv(csv_path, chunksize=batch_size)
    all_bin, all_3l, all_y = [], [], []
    n_total             = 0

    for i, chunk in enumerate(reader):
        X_chunk, y_chunk = preprocess(chunk, label_col, benign_label)
        p_bin, p_lab = _predict_one_mode(X_chunk, mode, monitor=None)
        all_bin.append(p_bin); all_3l.append(p_lab)
        if y_chunk is not None:
            all_y.append(y_chunk)
        n_total += len(X_chunk)
        if i % 10 == 0:
            print(f'  Batch {i+1:>4d}  processed={n_total:,}')

    all_bin = np.concatenate(all_bin)
    all_3l  = np.concatenate(all_3l)
    print(f'\n[done] {n_total:,} rows  Benign={int((all_bin==0).sum())}  '
          f'Malicious={int((all_bin==1).sum())}')

    results = {'predictions_binary': all_bin, 'predictions_3label': all_3l}
    if all_y:
        y_all = np.concatenate(all_y)
        print('\nBinary metrics:')
        results['metrics_binary'] = compute_metrics(y_all, all_bin, mode_name=f'{mode}-batch')
        print('3-label metrics:')
        results['metrics_3label'] = compute_metrics_3label(
            y_all, _to_code(all_3l), mode_name=f'{mode}-batch')
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='TRIDENT v13 Inference')
    parser.add_argument('--mode', choices=['a', 'b', 'c', 'd'], default='a',
                        help='Inference mode: a=CSV, b=single-row, c=all-modes, d=batch')
    parser.add_argument('--ckpt-dir', default=cfg.CKPT_DIR,
                        help='Checkpoint directory (overrides config.py)')
    args = parser.parse_args()

    print_system_info()
    load_artifacts(ckpt_dir=args.ckpt_dir)

    if args.mode == 'a':
        run_csv()
    elif args.mode == 'b':
        run_single_row()
    elif args.mode == 'c':
        run_all_modes()
    elif args.mode == 'd':
        run_batch_predictor()

    try:
        INFER_MONITOR.stop_all_polling()
    except Exception:
        pass
    INFER_MONITOR.print_full_report()


if __name__ == '__main__':
    main()
