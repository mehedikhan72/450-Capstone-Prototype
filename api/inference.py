#!/usr/bin/env python3
"""
VMFCVD Inference
================
Converted from vmfcvd-inference.ipynb.
Compatible with: vmfcvd_smote_3label_ham-et_RM (4-model FDM/DFDM, 5-model HAM)

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
    VMFCVD, FastDetectionMode, DefensiveFastDetectionMode,
    HighAccuracyMode, VMFCVDVoter, DetailedResourceMonitor,
    compute_metrics, compute_metrics_3label, print_system_info,
    USE_EARLY_EXIT,
)
import config as cfg

# ── Global session state ──────────────────────────────────────────────────────
INFER_MONITOR: DetailedResourceMonitor = None
VMFCVD_MODEL:  VMFCVD                  = None
SCALER                                  = None
ENCODERS                                = None
FDM_FEATURES                            = None
HAM_FEATURES                            = None
ALL_FEATURES                            = None


# ═════════════════════════════════════════════════════════════════════════════
# Artifact loading
# ═════════════════════════════════════════════════════════════════════════════

def _find_prefix(ckpt_dir: str) -> str:
    pattern = os.path.join(ckpt_dir, 'vmfcvd_*_step10_vmfcvd.pkl')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f'No step10 checkpoint in "{ckpt_dir}".\n'
            f'Run the training pipeline first so it saves vmfcvd_*_step10_vmfcvd.pkl.\n'
            f'Files found: {os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else "DIR NOT FOUND"}')
    if len(matches) > 1:
        names = [os.path.basename(m) for m in matches]
        raise RuntimeError(
            f'Multiple step10 checkpoints found: {names}.\n'
            f'Keep only the one you want in CKPT_DIR.')
    fname = os.path.basename(matches[0])
    return os.path.join(ckpt_dir, fname.split('_step10')[0])


def load_artifacts(ckpt_dir: str = cfg.CKPT_DIR) -> dict:
    global INFER_MONITOR, VMFCVD_MODEL, SCALER, ENCODERS
    global FDM_FEATURES, HAM_FEATURES, ALL_FEATURES

    INFER_MONITOR = DetailedResourceMonitor(poll_interval=0.05)
    prefix = _find_prefix(ckpt_dir)
    print(f'[ckpt] Using prefix: {os.path.basename(prefix)}')

    # vmfcvd object
    INFER_MONITOR.start('Load_vmfcvd_pkl')
    with open(f'{prefix}_step10_vmfcvd.pkl', 'rb') as f:
        vmfcvd = pickle.load(f)
    INFER_MONITOR.stop('Load_vmfcvd_pkl')
    print(f'[ckpt] vmfcvd      <- {os.path.basename(prefix)}_step10_vmfcvd.pkl')

    INFER_MONITOR.record_model_size('FDM_voter', vmfcvd.fdm.voter)
    INFER_MONITOR.record_model_size('HAM_voter', vmfcvd.ham.voter)

    # scaler + encoders
    INFER_MONITOR.start('Load_step03_pkl')
    with open(f'{prefix}_step03_transformed.pkl', 'rb') as f:
        _, scaler, encoders = pickle.load(f)
    INFER_MONITOR.stop('Load_step03_pkl')
    print(f'[ckpt] scaler/enc  <- {os.path.basename(prefix)}_step03_transformed.pkl')

    # feature lists
    INFER_MONITOR.start('Load_step09_pkl')
    with open(f'{prefix}_step09_clusters.pkl', 'rb') as f:
        _, fdm_features, ham_features = pickle.load(f)
    INFER_MONITOR.stop('Load_step09_pkl')
    print(f'[ckpt] features    <- {os.path.basename(prefix)}_step09_clusters.pkl')

    all_features = sorted(set(fdm_features) | set(ham_features))

    # Verify
    voter_fdm = vmfcvd.fdm.voter
    voter_ham = vmfcvd.ham.voter
    print(f'\n[verify] FDM voter:')
    print(f'  raw_models   : {list(voter_fdm.raw_models.keys())}  ({len(voter_fdm.raw_models)} models)')
    print(f'  MaxVoteIndex : {voter_fdm.MaxVoteIndex:.4f}')
    print(f'  Disagreement : {voter_fdm.disagreement_threshold:.4f}')
    print(f'[verify] HAM voter:')
    print(f'  raw_models   : {list(voter_ham.raw_models.keys())}  ({len(voter_ham.raw_models)} models)')
    print(f'  MaxVoteIndex : {voter_ham.MaxVoteIndex:.4f}')
    print(f'  Disagreement : {voter_ham.disagreement_threshold:.4f}')
    print(f'\n[features] FDM  : {fdm_features}')
    print(f'[features] HAM  : {ham_features}')
    print(f'[features] UNION: {all_features}  ← minimum columns your CSV needs')

    df_load = INFER_MONITOR.summary_df()
    if not df_load.empty:
        print('\n[resource] Loading stage times:')
        print(df_load[['Stage', 'Time (s)', 'Mem Δ (MB)', 'Peak Δ (MB)', 'Peak Mem (MB)']].to_string(index=False))

    # Assign to globals
    VMFCVD_MODEL = vmfcvd
    SCALER       = scaler
    ENCODERS     = encoders
    FDM_FEATURES = fdm_features
    HAM_FEATURES = ham_features
    ALL_FEATURES = all_features

    return dict(vmfcvd=vmfcvd, scaler=scaler, encoders=encoders,
                fdm_features=fdm_features, ham_features=ham_features,
                all_features=all_features)


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
            f'Your CSV must contain at least: {ALL_FEATURES}')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col, le in ENCODERS.items():
        if col in df.columns:
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

    X = df[ALL_FEATURES].astype(float)
    X.fillna(0, inplace=True)

    if SCALER is not None:
        scaler_feats  = list(SCALER.feature_names_in_)
        cols_to_scale = [c for c in ALL_FEATURES if c in scaler_feats]
        if cols_to_scale:
            idx = [scaler_feats.index(c) for c in cols_to_scale]
            X[cols_to_scale] = ((X[cols_to_scale].values
                                 - SCALER.mean_[idx]) / SCALER.scale_[idx])

    print(f'[preprocess] Ready: {X.shape[0]:,} rows × {X.shape[1]} features')
    return X, y


def single_row_to_df(row_dict: dict) -> pd.DataFrame:
    missing = [f for f in ALL_FEATURES if f not in row_dict]
    if missing:
        raise ValueError(f'SINGLE_ROW missing columns: {missing}\n'
                         f'Required: {ALL_FEATURES}')
    return pd.DataFrame([row_dict])[ALL_FEATURES]


def _flow_to_mode(flow_rate):
    if flow_rate is None:
        return None
    if flow_rate >= 5000:
        return 'DFDM'
    if flow_rate >= 1000:
        return 'FDM'
    return 'HAM'


def _predict_one_mode(X, mode_name: str, monitor=None):
    """Run binary + 3-label prediction for one mode, with optional resource tracking."""
    X_fdm = X[FDM_FEATURES].reset_index(drop=True)
    X_ham = X[HAM_FEATURES].reset_index(drop=True)

    if mode_name == 'FDM':
        if monitor:
            monitor.record_inference('FDM  binary',  VMFCVD_MODEL.fdm.predict,       X_fdm, cfg.N_REPEATS)
            monitor.record_inference('FDM  3-label', VMFCVD_MODEL.fdm.predict_3label, X_fdm, cfg.N_REPEATS)
        p_bin = VMFCVD_MODEL.fdm.predict(X_fdm)
        p_3l  = VMFCVD_MODEL.fdm.predict_3label(X_fdm)
    elif mode_name == 'DFDM':
        if monitor:
            monitor.record_inference('DFDM standard',   VMFCVD_MODEL.dfdm.predict,            X, cfg.N_REPEATS)
            monitor.record_inference('DFDM early-exit', VMFCVD_MODEL.dfdm.predict_early_exit, X, cfg.N_REPEATS)
        p_bin = VMFCVD_MODEL.dfdm.predict_early_exit(X) if USE_EARLY_EXIT else VMFCVD_MODEL.dfdm.predict(X)
        p_3l  = np.where(p_bin == 1, 2, 0)
    else:  # HAM
        if monitor:
            monitor.record_inference('HAM  binary',  VMFCVD_MODEL.ham.predict,       X_ham, cfg.N_REPEATS)
            monitor.record_inference('HAM  3-label', VMFCVD_MODEL.ham.predict_3label, X_ham, cfg.N_REPEATS)
        p_bin = VMFCVD_MODEL.ham.predict(X_ham)
        p_3l  = VMFCVD_MODEL.ham.predict_3label(X_ham)

    return p_bin, p_3l


# ═════════════════════════════════════════════════════════════════════════════
# Mode A — CSV inference
# ═════════════════════════════════════════════════════════════════════════════

def run_csv(csv_path=cfg.CSV_PATH, flow_rate=cfg.FLOW_RATE,
            label_col=cfg.LABEL_COL, benign_label=cfg.BENIGN_LABEL,
            save_output=True, track_resources=True):
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

    mode    = _flow_to_mode(flow_rate)
    modes   = [mode] if mode else ['HAM', 'FDM', 'DFDM']
    monitor = INFER_MONITOR if track_resources else None
    label3  = {0: 'Benign', 1: 'Warning', 2: 'Malicious'}
    all_res = {}

    for m in modes:
        print(f'\n{sep}\n  Predictions — {m}\n{sep}')
        t0 = time.perf_counter()
        p_bin, p_3l = _predict_one_mode(X, m, monitor=monitor)
        us = (time.perf_counter() - t0) / len(X) * 1e6
        n_mal = int((p_bin == 1).sum())
        print(f'[{m}] Benign={len(p_bin)-n_mal:,}  Malicious={n_mal:,}  '
              f'({n_mal/len(p_bin)*100:.2f}% attack)  {us:.2f} µs/sample (wall)')
        if m != 'DFDM':
            n_w = int((p_3l == 1).sum())
            print(f'[{m} 3L] Benign={int((p_3l==0).sum())}  Warning={n_w}  '
                  f'Malicious={int((p_3l==2).sum())}  '
                  f'(warn_rate={n_w/len(p_3l)*100:.2f}%)')
        res = {'predictions_binary': p_bin, 'predictions_3label': p_3l}
        if y is not None:
            print('\nBinary metrics:')
            res['metrics_binary'] = compute_metrics(y, p_bin, mode_name=m)
            if m != 'DFDM':
                print('3-label metrics:')
                res['metrics_3label'] = compute_metrics_3label(y, p_3l, mode_name=m)
        all_res[m] = res

    df_out = df_raw.copy()
    for m, r in all_res.items():
        sfx = '' if len(all_res) == 1 else f'_{m}'
        df_out[f'pred_binary{sfx}']  = r['predictions_binary']
        df_out[f'pred_3label{sfx}']  = pd.Series(r['predictions_3label']).map(label3).values
        df_out[f'mode{sfx}']         = m

    if save_output:
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
    label_3 = {0: 'Benign', 1: 'Warning',  2: 'Malicious'}

    for m in modes:
        p_bin, p_3l = _predict_one_mode(X, m, monitor=None)
        pred_b = label_b[int(p_bin[0])]
        pred_3 = label_3[int(p_3l[0])]
        correct = ''
        if true_label is not None:
            exp = label_b[int(true_label)]
            correct = ('  ✓ correct' if int(p_bin[0]) == int(true_label)
                       else f'  ✗ wrong (expected {exp})')
        print(f'\n  [{m}]')
        print(f'    Binary  : {pred_b}{correct}')
        if m != 'DFDM':
            print(f'    3-label : {pred_3}', end='')
            if pred_3 == 'Warning':
                print('  ← models disagree (high probability std)', end='')
            print()
        else:
            print('    DFDM is always binary (no warning zone in emergency mode)')

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
        p_bin, p_3l = _predict_one_mode(X, m, monitor=monitor)
        us  = (time.perf_counter() - t0) / len(X) * 1e6
        row = {'Mode': m, 'Speed_us(wall)': round(us, 2),
               'Malicious%': round(p_bin.mean() * 100, 2)}
        if y is not None:
            mb = compute_metrics(y, p_bin, mode_name=m)
            row.update({k: round(v, 6) for k, v in mb.items()
                        if k in ('Accuracy', 'Precision', 'Sensitivity', 'F1')})
            if m != 'DFDM':
                m3 = compute_metrics_3label(y, p_3l, mode_name=m)
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
                        benign_label=cfg.BENIGN_LABEL):
    """
    Mode D: mini-batch inference for files too large to load all at once.
    Slower than Mode A for normal-sized files — use only to avoid OOM.
    flow_rate must be a specific integer (not None) — one mode only.
    """
    if flow_rate is None:
        raise ValueError('Set FLOW_RATE to 0, 1000, or 5000 for batch mode.')
    mode = _flow_to_mode(flow_rate)
    print(f'  MODE D — Batch predictor  mode={mode}  batch_size={batch_size}')
    print('  NOTE: Use Mode A for normal files — it is faster.')

    reader              = pd.read_csv(csv_path, chunksize=batch_size)
    all_bin, all_3l, all_y = [], [], []
    n_total             = 0
    label3              = {0: 'Benign', 1: 'Warning', 2: 'Malicious'}

    for i, chunk in enumerate(reader):
        X_chunk, y_chunk = preprocess(chunk, label_col, benign_label)
        p_bin, p_3l = _predict_one_mode(X_chunk, mode, monitor=None)
        all_bin.append(p_bin); all_3l.append(p_3l)
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
        if mode != 'DFDM':
            print('3-label metrics:')
            results['metrics_3label'] = compute_metrics_3label(y_all, all_3l, mode_name=f'{mode}-batch')
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='VMFCVD Inference')
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
