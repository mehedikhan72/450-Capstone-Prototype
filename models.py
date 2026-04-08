# ── Standard library ─────────────────────────────────────────────────────────
import io, gc, pickle, threading, time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                               ExtraTreesClassifier,
                               GradientBoostingClassifier, RandomForestClassifier)

try:
    import psutil
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'psutil', '-q'])
    import psutil

# ── Training-time constants (must match training notebook exactly) ─────────────
N_ESTIMATORS            = 100
RANDOM_STATE            = 42
USE_CLASS_WEIGHTS       = True
USE_SOFT_VOTING         = True
USE_CALIBRATION         = True
USE_SHAPLEY_WEIGHTS     = True
USE_EXTRATREES_HAM_ONLY = True
USE_EARLY_EXIT          = True
WARNING_PENALTY         = 0.05
FLOW_THRESHOLD_HIGH     = 1_000
FLOW_THRESHOLD_EXTREME  = 5_000


# ═════════════════════════════════════════════════════════════════════════════
# DetailedResourceMonitor
# ═════════════════════════════════════════════════════════════════════════════

class DetailedResourceMonitor:
    def __init__(self, poll_interval: float = 0.05):
        self.process       = psutil.Process()
        self.poll_interval = poll_interval
        self.stages: Dict[str, dict] = {}
        self.model_sizes: Dict[str, int] = {}
        self.inference_results: Dict[str, dict] = {}
        self._lock            = threading.Lock()
        self._active_stage    = None
        self._stop_event      = threading.Event()
        self._poll_thread     = threading.Thread(
            target=self._poll_worker, daemon=True, name='resource-monitor-poll')
        self._poll_thread.start()
        self._mem_samples:    Dict[str, List[float]] = {}
        self._cpu_samples:    Dict[str, List[float]] = {}
        self._thread_samples: Dict[str, List[int]]   = {}

    def _poll_worker(self):
        while not self._stop_event.is_set():
            with self._lock:
                stage = self._active_stage
            if stage:
                try:
                    mem_mb  = self.process.memory_info().rss / 1024 / 1024
                    cpu_pct = self.process.cpu_percent()
                    n_thr   = self.process.num_threads()
                    self._mem_samples[stage].append(mem_mb)
                    self._cpu_samples[stage].append(cpu_pct)
                    self._thread_samples[stage].append(n_thr)
                except Exception:
                    pass
            time.sleep(self.poll_interval)

    def start(self, stage: str):
        gc.collect()
        mem_now = self.process.memory_info().rss / 1024 / 1024
        self.stages[stage] = {
            'start_time'    : time.perf_counter(),
            'start_mem_mb'  : mem_now,
            'start_threads' : self.process.num_threads(),
        }
        self._mem_samples[stage]    = [mem_now]
        self._cpu_samples[stage]    = []
        self._thread_samples[stage] = [self.process.num_threads()]
        with self._lock:
            self._active_stage = stage

    def stop(self, stage: str) -> dict:
        with self._lock:
            if self._active_stage == stage:
                self._active_stage = None
        if stage not in self.stages:
            return {}
        s = self.stages[stage]
        s['elapsed_sec']      = time.perf_counter() - s['start_time']
        s['end_mem_mb']       = self.process.memory_info().rss / 1024 / 1024
        s['end_threads']      = self.process.num_threads()
        mem_s = self._mem_samples.get(stage, [s['end_mem_mb']])
        cpu_s = self._cpu_samples.get(stage, [0.0])
        thr_s = self._thread_samples.get(stage, [s['start_threads']])
        s['peak_mem_mb']      = float(max(mem_s))
        s['min_mem_mb']       = float(min(mem_s))
        s['mem_delta_mb']     = s['end_mem_mb'] - s['start_mem_mb']
        s['peak_increase_mb'] = s['peak_mem_mb'] - s['start_mem_mb']
        s['avg_cpu_pct']      = float(np.mean(cpu_s)) if cpu_s else 0.0
        s['max_cpu_pct']      = float(np.max(cpu_s))  if cpu_s else 0.0
        s['avg_threads']      = float(np.mean(thr_s))
        s['max_threads']      = int(np.max(thr_s))
        s['n_poll_samples']   = len(mem_s)
        return s

    def stop_all_polling(self):
        self._stop_event.set()
        self._poll_thread.join(timeout=2.0)

    def record_model_size(self, label: str, model) -> int:
        try:
            buf = io.BytesIO()
            pickle.dump(model, buf, protocol=4)
            nbytes = buf.tell()
        except Exception:
            nbytes = 0
        self.model_sizes[label] = nbytes
        return nbytes

    def record_inference(self, label: str, predict_fn, X,
                         n_repeats: int = 5, warmup: int = 1):
        n = len(X)
        for _ in range(warmup):
            predict_fn(X)
        stage = f'_infer_{label}'
        self.start(stage)
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            predict_fn(X)
            times.append(time.perf_counter() - t0)
        self.stop(stage)
        mean_sec = float(np.mean(times))
        std_sec  = float(np.std(times))
        s = self.stages[stage]
        self.inference_results[label] = {
            'per_sample_us'    : mean_sec / n * 1e6,
            'per_sample_std_us': std_sec  / n * 1e6,
            'total_ms'         : mean_sec * 1000,
            'throughput_sps'   : n / mean_sec if mean_sec > 0 else 0,
            'peak_mem_mb'      : s['peak_mem_mb'],
            'mem_delta_mb'     : s['mem_delta_mb'],
            'avg_cpu_pct'      : s['avg_cpu_pct'],
            'max_cpu_pct'      : s['max_cpu_pct'],
            'max_threads'      : s['max_threads'],
            'n_samples'        : n,
            'n_repeats'        : n_repeats,
        }
        return self.inference_results[label]

    def summary_df(self, exclude_prefix: str = '_infer_') -> pd.DataFrame:
        rows = []
        for name, s in self.stages.items():
            if name.startswith(exclude_prefix) or 'elapsed_sec' not in s:
                continue
            rows.append({
                'Stage'         : name,
                'Time (s)'      : round(s['elapsed_sec'], 3),
                'Mem Δ (MB)'    : round(s['mem_delta_mb'], 2),
                'Peak Δ (MB)'   : round(s['peak_increase_mb'], 2),
                'Peak Mem (MB)' : round(s['peak_mem_mb'], 2),
                'Avg CPU (%)'   : round(s['avg_cpu_pct'], 1),
                'Max CPU (%)'   : round(s['max_cpu_pct'], 1),
                'Max Threads'   : s['max_threads'],
                'Poll samples'  : s['n_poll_samples'],
            })
        return pd.DataFrame(rows)

    def print_inference_report(self):
        if not self.inference_results:
            print('  No inference results recorded yet.')
            return
        print('\n' + '='*65)
        print('  INFERENCE RESOURCE BREAKDOWN — per prediction path')
        print('='*65)
        hdr = (f'  {"Path":<28}  {"µs/sample":>9}  {"±std µs":>7}  '
               f'{"throughput/s":>13}  {"Peak ΔMem MB":>12}  {"Avg CPU%":>8}  {"MaxThr":>6}')
        print(hdr)
        print('  ' + '-'*92)
        for label, r in self.inference_results.items():
            print(f'  {label:<28}  {r["per_sample_us"]:>9.3f}  '
                  f'{r["per_sample_std_us"]:>7.3f}  '
                  f'{r["throughput_sps"]:>13,.0f}  '
                  f'{r["mem_delta_mb"]:>+12.2f}  '
                  f'{r["avg_cpu_pct"]:>8.1f}  '
                  f'{r["max_threads"]:>6}')

    def print_stage_summary(self):
        df = self.summary_df()
        if df.empty:
            print('  No stages recorded.')
            return
        print('\n' + '='*65)
        print('  STAGE RESOURCE SUMMARY')
        print('='*65)
        print(df.to_string(index=False))

    def print_fdm_vs_ham_delta(self):
        print('\n' + '='*65)
        print('  FDM (4 models) vs HAM (5 models) — inference resource delta')
        print('  ExtraTrees is HAM-only; this shows the cost.')
        print('='*65)
        for tag, fdm_key, ham_key in [
            ('binary',  'FDM  binary',  'HAM  binary'),
            ('3-label', 'FDM  3-label', 'HAM  3-label'),
        ]:
            if fdm_key in self.inference_results and ham_key in self.inference_results:
                fu = self.inference_results[fdm_key]['per_sample_us']
                hu = self.inference_results[ham_key]['per_sample_us']
                print(f'\n  {tag}:')
                print(f'    FDM (4m) : {fu:.3f} µs/sample')
                print(f'    HAM (5m) : {hu:.3f} µs/sample')
                print(f'    Delta    : {hu-fu:+.3f} µs  ({(hu/fu-1)*100:+.1f}%)')

    def print_efficiency_table(self):
        if not self.inference_results:
            return
        print('\n' + '='*65)
        print('  EFFICIENCY METRICS  (higher = better use of resources)')
        print('='*65)
        rows = []
        for label, r in self.inference_results.items():
            us   = r['per_sample_us']
            sps  = r['throughput_sps']
            peak = max(abs(r['mem_delta_mb']), 0.1)
            rows.append({
                'Path'              : label,
                'Samples/s per MB'  : round(sps / peak, 1),
                'Samples/s per CPU%': round(sps / max(r['avg_cpu_pct'], 0.1), 0),
                'µs × Peak ΔMB'    : round(us * peak, 4),
            })
        print(pd.DataFrame(rows).to_string(index=False))

    def print_full_report(self):
        print('\n' + '#'*65)
        print('  FULL INFERENCE RESOURCE REPORT')
        print('#'*65)
        self.print_stage_summary()
        self.print_inference_report()
        self.print_fdm_vs_ham_delta()
        self.print_efficiency_table()


# Alias for backward compat
ResourceMonitor = DetailedResourceMonitor


# ═════════════════════════════════════════════════════════════════════════════
# VMFCVDVoter
# ═════════════════════════════════════════════════════════════════════════════

class VMFCVDVoter:
    _BASE_MODELS_SPEC = {
        'AdaBoost'        : lambda: AdaBoostClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
        'Bagging'         : lambda: BaggingClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
        'RandomForest'    : lambda: RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
    }
    _EXTRATREES_SPEC = {
        'ExtraTrees': lambda: ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
    }

    def __init__(self, use_class_weights=USE_CLASS_WEIGHTS,
                 use_soft_voting=USE_SOFT_VOTING, use_calibration=USE_CALIBRATION,
                 use_shapley=USE_SHAPLEY_WEIGHTS, include_extratrees=False):
        self.include_extratrees = include_extratrees
        self._models_spec = dict(self._BASE_MODELS_SPEC)
        if include_extratrees:
            self._models_spec['ExtraTrees'] = self._EXTRATREES_SPEC['ExtraTrees']
        self.raw_models = {}
        self.trained_models = {}
        self.model_accuracies = {}
        self.model_weights = {}
        self.TotAccuracy = None
        self.MaxVoteIndex = None
        self.MaxAccuracy = None
        self.disagreement_threshold = None
        self.use_class_weights = use_class_weights
        self.use_soft_voting   = use_soft_voting
        self.use_calibration   = use_calibration
        self.use_shapley       = use_shapley

    def voting_data(self, X):
        VD = np.zeros(len(X), dtype=float)
        for name, model in self.raw_models.items():
            VD += model.predict(X).astype(float) * self.model_weights.get(name, 1.0)
        return VD

    def hard_voting_data(self, X):
        VD = np.zeros(len(X), dtype=float)
        for model in self.raw_models.values():
            VD += model.predict(X).astype(float)
        return VD

    def predict_binary_from_vd(self, VD):
        return (VD >= self.MaxVoteIndex).astype(int)

    def predict_3label(self, X):
        probas = np.stack(
            [m.predict_proba(X)[:, 1] for m in self.trained_models.values()], axis=1)
        mean_p = probas.mean(axis=1)
        std_p  = probas.std(axis=1)
        return np.where(std_p >= self.disagreement_threshold, 1,
               np.where(mean_p >= 0.5, 2, 0))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_models_spec', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._models_spec = dict(self._BASE_MODELS_SPEC)
        if getattr(self, 'include_extratrees', False):
            self._models_spec['ExtraTrees'] = self._EXTRATREES_SPEC['ExtraTrees']


# ═════════════════════════════════════════════════════════════════════════════
# FastDetectionMode
# ═════════════════════════════════════════════════════════════════════════════

class FastDetectionMode:
    def __init__(self, fdm_features: list):
        self.fdm_features = fdm_features
        self.voter        = VMFCVDVoter(include_extratrees=False)
        self.trained      = False

    def predict(self, X) -> np.ndarray:
        VD = self.voter.voting_data(X[self.fdm_features])
        return (VD >= self.voter.MaxVoteIndex).astype(int)

    def predict_3label(self, X) -> np.ndarray:
        return self.voter.predict_3label(X[self.fdm_features])

    def evaluate(self, X_test, y_test) -> dict:
        return compute_metrics(np.array(y_test), self.predict(X_test), mode_name='FDM')

    def evaluate_3label(self, X_test, y_test) -> dict:
        return compute_metrics_3label(np.array(y_test),
                                      self.predict_3label(X_test), mode_name='FDM')


# ═════════════════════════════════════════════════════════════════════════════
# DefensiveFastDetectionMode
# ═════════════════════════════════════════════════════════════════════════════

class DefensiveFastDetectionMode:
    def __init__(self, fdm: FastDetectionMode):
        self.fdm = fdm

    @property
    def trained(self):
        return self.fdm.trained

    def predict(self, X) -> np.ndarray:
        VD = self.fdm.voter.hard_voting_data(X[self.fdm.fdm_features])
        return (VD > 0).astype(int)

    def predict_early_exit(self, X) -> np.ndarray:
        X_feat       = X[self.fdm.fdm_features].reset_index(drop=True)
        n            = len(X_feat)
        results      = np.zeros(n, dtype=int)
        still_benign = np.ones(n, dtype=bool)
        for name in list(self.fdm.voter.raw_models.keys()):
            if not still_benign.any():
                break
            model       = self.fdm.voter.raw_models[name]
            benign_idx  = np.where(still_benign)[0]
            X_remaining = X_feat.iloc[benign_idx].reset_index(drop=True)
            preds       = model.predict(X_remaining)
            flagged     = benign_idx[preds.astype(bool)]
            results[flagged]      = 1
            still_benign[flagged] = False
        return results

    def evaluate(self, X_test, y_test) -> dict:
        return compute_metrics(np.array(y_test), self.predict(X_test), mode_name='DFDM')

    def evaluate_early_exit(self, X_test, y_test) -> dict:
        return compute_metrics(np.array(y_test),
                               self.predict_early_exit(X_test), mode_name='DFDM-EarlyExit')


# ═════════════════════════════════════════════════════════════════════════════
# HighAccuracyMode
# ═════════════════════════════════════════════════════════════════════════════

class HighAccuracyMode:
    def __init__(self, ham_features: list):
        self.ham_features = ham_features
        self.voter        = VMFCVDVoter(include_extratrees=USE_EXTRATREES_HAM_ONLY)
        self.trained      = False

    def predict(self, X) -> np.ndarray:
        VD = self.voter.voting_data(X[self.ham_features])
        return (VD >= self.voter.MaxVoteIndex).astype(int)

    def predict_3label(self, X) -> np.ndarray:
        return self.voter.predict_3label(X[self.ham_features])

    def evaluate(self, X_test, y_test) -> dict:
        return compute_metrics(np.array(y_test), self.predict(X_test), mode_name='HAM')

    def evaluate_3label(self, X_test, y_test) -> dict:
        return compute_metrics_3label(np.array(y_test),
                                      self.predict_3label(X_test), mode_name='HAM')


# ═════════════════════════════════════════════════════════════════════════════
# VMFCVD controller
# ═════════════════════════════════════════════════════════════════════════════

class VMFCVD:
    def __init__(self, fdm_features: list, ham_features: list,
                 flow_threshold_high: int   = FLOW_THRESHOLD_HIGH,
                 flow_threshold_extreme: int = FLOW_THRESHOLD_EXTREME):
        self.fdm  = FastDetectionMode(fdm_features)
        self.dfdm = DefensiveFastDetectionMode(self.fdm)
        self.ham  = HighAccuracyMode(ham_features)
        self.flow_threshold_high    = flow_threshold_high
        self.flow_threshold_extreme = flow_threshold_extreme
        self.current_mode  = 'HAM'
        self.trained       = False
        self.smote_applied = False

    def _select_mode(self, network_flow_rate: int) -> str:
        if network_flow_rate >= self.flow_threshold_extreme:
            mode = 'DFDM'
        elif network_flow_rate >= self.flow_threshold_high:
            mode = 'FDM'
        else:
            mode = 'HAM'
        self.current_mode = mode
        return mode

    def predict(self, X, network_flow_rate: int = 0) -> tuple:
        mode = self._select_mode(network_flow_rate)
        if mode == 'DFDM':
            return (self.dfdm.predict_early_exit(X) if USE_EARLY_EXIT
                    else self.dfdm.predict(X)), mode
        elif mode == 'FDM':
            return self.fdm.predict(X), mode
        else:
            return self.ham.predict(X), mode

    def predict_3label(self, X, network_flow_rate: int = 0) -> tuple:
        mode = self._select_mode(network_flow_rate)
        if mode == 'DFDM':
            binary = self.dfdm.predict(X)
            return np.where(binary == 1, 2, 0), mode
        elif mode == 'FDM':
            return self.fdm.predict_3label(X), mode
        else:
            return self.ham.predict_3label(X), mode

    def evaluate_all_modes(self, X_test, y_test) -> dict:
        print('\n' + '='*65 + '\n  VMFCVD -- Binary Evaluation\n' + '='*65)
        return {
            'FDM' : self.fdm.evaluate(X_test, y_test),
            'DFDM': self.dfdm.evaluate(X_test, y_test),
            'HAM' : self.ham.evaluate(X_test, y_test),
        }

    def individual_model_accuracies(self) -> pd.DataFrame:
        rows = []
        for mode_name, voter in [('FDM/DFDM', self.fdm.voter), ('HAM', self.ham.voter)]:
            for model_name, acc in voter.model_accuracies.items():
                rows.append({'Mode': mode_name, 'Model': model_name, 'Accuracy': acc})
        return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, mode_name='') -> dict:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    TP = int(((y_true==1)&(y_pred==1)).sum())
    TN = int(((y_true==0)&(y_pred==0)).sum())
    FP = int(((y_true==0)&(y_pred==1)).sum())
    FN = int(((y_true==1)&(y_pred==0)).sum())
    total = TP+TN+FP+FN
    acc  = (TP+TN)/total        if total    else 0.
    prec = TP/(TP+FP)           if TP+FP    else 0.
    rec  = TP/(TP+FN)           if TP+FN    else 0.
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.
    tag  = f'[{mode_name}] ' if mode_name else ''
    print(f'{tag}Accuracy={acc:.6f}  Precision={prec:.6f}  '
          f'Sensitivity={rec:.6f}  F1={f1:.6f}')
    print(f'{tag}TP={TP}  TN={TN}  FP={FP}  FN={FN}')
    return dict(Accuracy=acc, Precision=prec, Sensitivity=rec, F1=f1,
                TP=TP, TN=TN, FP=FP, FN=FN)


def compute_metrics_3label(y_true, y_pred_3, mode_name='') -> dict:
    y_true   = np.asarray(y_true).flatten()
    y_pred_3 = np.asarray(y_pred_3).flatten()
    n        = len(y_true)
    b = (y_pred_3==0); w = (y_pred_3==1); m = (y_pred_3==2)
    conf = ~w
    conf_acc = 0.
    if conf.sum():
        yb = np.where(y_pred_3==2, 1, 0)
        conf_acc = metrics.accuracy_score(y_true[conf], yb[conf])
    mal_tp   = int(((y_true==1)&m).sum())
    ben_tp   = int(((y_true==0)&b).sum())
    ben_fp   = int(((y_true==1)&b).sum())
    warn_b   = int(((y_true==0)&w).sum())
    warn_m   = int(((y_true==1)&w).sum())
    mal_rec  = mal_tp / max(int((y_true==1).sum()), 1)
    ben_rec  = ben_tp / max(int((y_true==0).sum()), 1)
    ben_prec = ben_tp / max(int(b.sum()), 1)
    mal_prec = mal_tp / max(int(m.sum()), 1)
    wr = w.sum()/n
    tag = f'[{mode_name} 3L] ' if mode_name else ''
    print(f'{tag}ConfidentAcc={conf_acc:.6f}  WarningRate={wr:.4f}  '
          f'MalRecall={mal_rec:.6f}  BenRecall={ben_rec:.6f}')
    print(f'{tag}  Counts: benign={b.sum()}  warning={w.sum()}  malicious={m.sum()}')
    print(f'{tag}  Warnings: {warn_b} benign + {warn_m} malicious')
    print(f'{tag}  AttacksLetThrough={ben_fp}')
    return dict(ConfidentAccuracy=conf_acc, WarningRate=float(wr),
                BenignPrecision=ben_prec, BenignRecall=ben_rec,
                MaliciousPrecision=mal_prec, MaliciousRecall=mal_rec,
                AttacksLetThrough=ben_fp,
                Counts=dict(benign=int(b.sum()), warning=int(w.sum()), malicious=int(m.sum())),
                WarningBreakdown=dict(actually_benign=warn_b, actually_malicious=warn_m))


def print_system_info():
    import sys
    vm = psutil.virtual_memory()
    print('\n' + '='*65)
    print('  SYSTEM INFORMATION')
    print('='*65)
    print(f'  CPU cores  (physical) : {psutil.cpu_count(logical=False)}')
    print(f'  CPU cores  (logical)  : {psutil.cpu_count(logical=True)}')
    print(f'  Total RAM             : {vm.total / 1024**3:.2f} GB')
    print(f'  Available RAM         : {vm.available / 1024**3:.2f} GB')
    print(f'  RAM utilisation       : {vm.percent:.1f}%')
    print(f'  Python                : {sys.version.split()[0]}')
    print(f'  Platform              : {sys.platform}')
