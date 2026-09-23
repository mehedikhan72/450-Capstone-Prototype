"""
TRIDENT v13 object graph — the classes the trained bundle unpickles into.
================================================================================
Ported from Notebooks/14-trident-final.ipynb (build v13.1). The five classes and
the two module-level constants below are copied VERBATIM from that notebook and
must stay that way: the .joblib holds instance state only, so these definitions
are what give it behaviour.

THE TWO CONSTANTS ARE PART OF THE MODEL, NOT CONFIGURATION.
    Trident.score() and Trident.predict_deploy() read HEADLINE_PATH and
    DEFAULT_PATH as module globals. Omit them and prediction raises; change their
    values and the service runs happily while answering differently from the
    reported results -- HAM through the stacker (2x slower, same output) and FDM
    through the vote (one fewer attack caught per test split).

The V6 object graph is preserved in models_v6_legacy.py.
"""

# ── Standard library ─────────────────────────────────────────────────────────
import io, gc, math, pickle, threading, time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:                      # the bundle WILL fail to unpickle without it:
    lgb, HAS_LGB = None, False           # the 'Boosting' member is an LGBMClassifier

try:
    import psutil
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'psutil', '-q'])
    import psutil


# ── Training-time constants (must match the training notebook exactly) ───────
RANDOM_STATE            = 42
N_ESTIMATORS            = 100
USE_CALIBRATION         = True
USE_SHAPLEY             = True
USE_STACKING            = True
USE_LINREG_IMPUTE       = True           # VMFCVD Algorithm 1
CALIBRATION_MODE        = 'oob'          # 'oob' | 'cv3'
CALIBRATION_MIN_OOB     = 200
MIN_ENSEMBLE_SIZE       = 4
MAXVOTE_STEPS           = 41
DIVERSITY_SUBSAMPLE     = 0.80
DIVERSITY_FEATFRAC      = 0.90
DIVERSITY_FEATFRAC_BIG  = 0.70
WARNING_PENALTY         = 0.05
WARN_RATE_FLOOR         = 0.0005
WARN_RATE_CEIL          = 0.02
FLOW_THRESHOLD_HIGH     = 1_000          # flows/s above which FDM replaces HAM
FLOW_THRESHOLD_EXTREME  = 5_000          # flows/s above which DFDM replaces FDM
USE_EARLY_EXIT          = True           # DFDM cascade; kept for inference.py

# ── PART OF THE MODEL. See the module docstring. ─────────────────────────────
HEADLINE_PATH = {'FDM': 'stack', 'HAM': 'vote'}
DEFAULT_PATH  = 'vote'


# ═════════════════════════════════════════════════════════════════════════════
# TRIDENT v13 classes — verbatim from the training notebook
# ═════════════════════════════════════════════════════════════════════════════


class LinRegImputer:
    '''VMFCVD Algorithm 1. Per column with gaps, regress on its most-correlated
    fully-observed donor column; fall back to the training median if no usable donor.'''
    def __init__(self, cols): self.cols = list(cols)
    def fit(self, df):
        X = df[self.cols]
        self.median_ = X.median(numeric_only=True)
        self.models_ = {}
        gappy = [c for c in self.cols if X[c].isna().any()]
        full  = [c for c in self.cols if not X[c].isna().any() and X[c].nunique() > 1]
        if full:
            corr = X[gappy + full].corr(numeric_only=True).abs()
            for c in gappy:
                donors = corr.loc[c, full].dropna().sort_values(ascending=False)
                if len(donors) and donors.iloc[0] > 0.30:
                    dcol = donors.index[0]
                    ok = X[c].notna()
                    lr = LinearRegression().fit(X.loc[ok, [dcol]], X.loc[ok, c])
                    self.models_[c] = (dcol, lr, float(donors.iloc[0]))
        self.gappy_ = gappy
        return self
    def transform(self, df):
        X = df[self.cols].copy()
        for c in self.cols:
            if not X[c].isna().any(): continue
            m = X[c].isna()
            if c in self.models_:
                dcol, lr, _ = self.models_[c]
                X.loc[m, c] = lr.predict(X.loc[m, [dcol]].fillna(self.median_[dcol]))
            else:
                X.loc[m, c] = self.median_[c]
        return X
    def report(self):
        if not self.gappy_: return '  no missing values to impute'
        L = []
        for c in self.gappy_:
            if c in self.models_:
                d, _, r = self.models_[c]; L.append(f'  {c:26s} <- linreg on {d} (|r|={r:.3f})')
            else: L.append(f'  {c:26s} <- training median (no donor with |r|>0.30)')
        return '\n'.join(L)


class Pre:
    def fit(self, df, feats):
        self.cols = list(feats)
        self.imp = LinRegImputer(self.cols).fit(df) if USE_LINREG_IMPUTE else None
        self.med = df[self.cols].median(numeric_only=True)
        base = self.imp.transform(df) if self.imp else df[self.cols].fillna(self.med)
        self.pt = PowerTransformer('yeo-johnson', standardize=True).fit(base)
        return self
    def transform(self, df):
        base = self.imp.transform(df) if self.imp else df[self.cols].fillna(self.med)
        return pd.DataFrame(self.pt.transform(base), columns=self.cols, index=df.index)


def make_model(name, seed, n_est=N_ESTIMATORS):
    s = seed
    if name=='AdaBoost':
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=s),
                                  n_estimators=max(30,n_est//2), learning_rate=0.5, random_state=s)
    if name=='Bagging':
        return BaggingClassifier(estimator=DecisionTreeClassifier(random_state=s),
                                 n_estimators=max(20,n_est//4), n_jobs=-1, random_state=s)
    if name=='Boosting':
        return (lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.1, num_leaves=31,
                                   class_weight='balanced', random_state=s, n_jobs=-1, verbose=-1)
                if HAS_LGB else GradientBoostingClassifier(n_estimators=max(40,n_est//2), random_state=s))
    if name=='RandomForest':
        return RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=s,
                                      class_weight='balanced_subsample')
    if name=='ExtraTrees':
        return ExtraTreesClassifier(n_estimators=n_est, n_jobs=-1, random_state=s,
                                    class_weight='balanced')
    if name=='DecisionTree':
        return DecisionTreeClassifier(random_state=s, class_weight='balanced')
    raise ValueError(name)


class Member:
    '''One ensemble member: own seed, own row bootstrap, own feature subspace, own calibrator.'''
    def __init__(self, name, feats, seed, n_est=N_ESTIMATORS, params=None):
        self.name, self.seed, self.n_est, self.params = name, seed, n_est, dict(params or {})
        rng = np.random.RandomState(seed)
        frac = DIVERSITY_FEATFRAC if len(feats) <= 12 else DIVERSITY_FEATFRAC_BIG
        if len(feats) > 2 and frac < 1.0:
            k = max(2, int(round(len(feats)*frac)))
            self.feats = sorted(rng.choice(feats, k, replace=False).tolist(), key=list(feats).index)
        else:
            self.feats = list(feats)
        self.clf = self.cal = None; self.fit_s = 0.0; self.cost_us = np.inf; self.size_kb = 0.0
        self.cal_mode = 'none'
    def _new(self):
        m = make_model(self.name, self.seed, self.n_est)
        if self.params:
            try: m.set_params(**{k:v for k,v in self.params.items() if k in m.get_params()})
            except Exception: pass
        return m
    def _calibrate(self, X, y, Xs, ys, oob):
        '''Prefer OUT-OF-BAG: one calibrator fitted on rows the base model never saw. cv=3
        averages three sub-models, so the calibrated path costs three times as much at
        inference, and it calibrates on the same bootstrap the model was fitted to. Falls back
        to cv=3 when the OOB set is too small or single-class, or when the installed sklearn
        will not accept a prefit estimator.'''
        if (CALIBRATION_MODE == 'oob' and len(oob) >= CALIBRATION_MIN_OOB
                and len(np.unique(y[oob])) > 1):
            try:
                c = CalibratedClassifierCV(self.clf, method='sigmoid', cv='prefit')
                c.fit(X.iloc[oob][self.feats], y[oob])
                return c, 'oob'
            except Exception:
                pass
        try:
            return CalibratedClassifierCV(self._new(), method='sigmoid', cv=3).fit(Xs, ys), 'cv3'
        except Exception:
            return None, 'none'

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(X), int(len(X)*DIVERSITY_SUBSAMPLE), replace=True)
        oob = np.setdiff1d(np.arange(len(X)), np.unique(idx))
        Xs, ys = X.iloc[idx][self.feats], y[idx]
        if len(np.unique(ys)) < 2: Xs, ys, oob = X[self.feats], y, np.arange(0)
        t0 = time.perf_counter()
        self.clf = self._new().fit(Xs, ys)
        if USE_CALIBRATION:
            self.cal, self.cal_mode = self._calibrate(X, y, Xs, ys, oob)
        self.fit_s = time.perf_counter()-t0
        try: self.size_kb = len(pickle.dumps(self.clf))/1024
        except Exception: pass
        return self
    def proba(self, X):  return (self.cal or self.clf).predict_proba(X[self.feats])[:,1]
    def predict(self, X): return self.clf.predict(X[self.feats]).astype(int)   # raw = cheap path


def shapley_weights(members, X, y, n_perm=60, seed=RANDOM_STATE):
    '''Coalition value is scored on HARD VOTES, because hard votes are what the weights are
    then applied to (Algorithm 3 sums weighted votes against MaxVoteIndex). v10-v12 scored it
    on averaged CALIBRATED PROBABILITIES instead -- a regression introduced when the voter was
    rewritten; V6 used raw votes on both sides. Fitting the weights against a different rule
    from the one they serve made them right only by correlation.'''
    V = np.column_stack([m.predict(X) for m in members]); n = len(members)
    cache = {}
    def v(S):
        k = tuple(sorted(S))
        if k not in cache:
            cache[k] = 0.0 if not k else metrics.matthews_corrcoef(y, (V[:,list(k)].mean(1)>=0.5).astype(int))
        return cache[k]
    rng = np.random.RandomState(seed); phi = np.zeros(n)
    for _ in range(n_perm):
        S, prev = [], 0.0
        for i in rng.permutation(n):
            S.append(i); cur = v(S); phi[i] += cur - prev; prev = cur
    phi = np.maximum(phi/n_perm, 0.0)
    return phi/phi.sum() if phi.sum() > 0 else np.ones(n)/n


class Trident:
    def __init__(self, names, feats, tag, tuned=None, seed_base=RANDOM_STATE, mode=None):
        T = tuned or {}
        self.tag, self.feats = tag, list(feats)
        # self.mode drives the HEADLINE_PATH lookup, so it must be EXACTLY 'FDM' or 'HAM'.
        # Deriving it from the display tag was a live defect: the replication loop tags its
        # ensembles 'FDM-s42', which matches no key, silently fell through to DEFAULT_PATH
        # ('vote') and scored every FDM replication on a different path from the FDM headline
        # (seed 42 read 0.991349 against the headline's 0.991428 -- the same models, two paths).
        # Callers that are not a headline mode now say so explicitly; the tag stays for display.
        self.mode = mode or tag.split('/')[-1]
        self.members = [Member(nm, feats, seed_base + 101*i,
                               T.get(nm,{}).get('n_estimators', N_ESTIMATORS), T.get(nm))
                        for i, nm in enumerate(names)]
        self.w = self.meta = None
        self.maxvote = 0.5; self.dissent_min = np.inf; self.tau = np.inf
        self.maxvote_tied_range = (0.5, 0.5, 1)

    def fit(self, Xt, yt):
        for m in self.members: m.fit(Xt, yt)
        assert len(self.members) >= MIN_ENSEMBLE_SIZE, \
            f'{self.tag}: {len(self.members)} members < {MIN_ENSEMBLE_SIZE} — V8 regression'
        return self

    # --- Algorithm 3 --------------------------------------------------------
    def _P(self, X): return np.column_stack([m.proba(X)   for m in self.members])
    def _V(self, X): return np.column_stack([m.predict(X) for m in self.members])
    @staticmethod
    def _packed(X, members):
        '''Slice the frame ONCE into the union of member columns, then index with numpy.
        Repeated DataFrame row+column slicing per member costs more than these models do,
        which is what made the first DFDM cascade measure slower than no cascade at all.'''
        cols = list(dict.fromkeys(f for m in members for f in m.feats))
        M = X[cols].to_numpy()
        pos = {c: i for i, c in enumerate(cols)}
        return M, [np.array([pos[f] for f in m.feats]) for m in members]
    def votes_fast(self, X):
        '''Hard votes from every member with no early exit — the DFDM speedup baseline.'''
        M, mi = self._packed(X, self.members)
        return np.column_stack([m.clf.predict(M[:, ix]) for m, ix in zip(self.members, mi)])
    def voting_data(self, X): return self._V(X) @ self.w          # VD in [0,1]
    def predict_from_vd(self, VD): return (VD >= self.maxvote).astype(int)

    def calibrate(self, Xv, yv, gv):
        A = (np.asarray(gv) % 2 == 0); B = ~A
        if A.sum() < 50 or B.sum() < 50: A = B = np.ones(len(yv), bool)
        Xa, ya, Xb, yb = Xv[A], yv[A], Xv[B], yv[B]
        self.w = (shapley_weights(self.members, Xa, ya) if USE_SHAPLEY
                  else np.ones(len(self.members))/len(self.members))
        VDa = self.voting_data(Xa)
        grid = np.linspace(0.05, 0.95, MAXVOTE_STEPS)
        sc = np.array([metrics.matthews_corrcoef(ya, (VDa >= t).astype(int)) for t in grid])
        # When members agree, VD is almost always 0 or 1 and a wide band of thresholds ties.
        # argmax would return the lowest tied value -- an artefact that reads as a degenerate
        # quorum. Take the MIDPOINT of the tied band: the most robust point, not the first.
        tied = grid[sc >= sc.max() - 1e-12]
        self.maxvote = float(np.median(tied))
        self.maxvote_tied_range = (float(tied.min()), float(tied.max()), int(len(tied)))
        if USE_STACKING:
            self.meta = LogisticRegression(max_iter=2000, class_weight='balanced')\
                        .fit(np.hstack([self._P(Xa), self._V(Xa)]), ya)
        self._fit_warning(Xb, yb)
        return self

    def score(self, X, use_stack=None):
        P, V = self._P(X), self._V(X)
        VD = V @ self.w
        p_vote, maj_vote = P @ self.w, (VD >= self.maxvote).astype(int)
        if self.meta is not None:
            p_stack = self.meta.predict_proba(np.hstack([P, V]))[:, 1]
            maj_stack = (p_stack >= 0.5).astype(int)
        else:
            p_stack, maj_stack = p_vote, maj_vote
        if use_stack is None:
            use_stack = (HEADLINE_PATH.get(self.mode, DEFAULT_PATH) == 'stack')
        p, maj = (p_stack, maj_stack) if use_stack else (p_vote, maj_vote)
        dissent = (V != maj[:, None]).astype(float) @ self.w
        return p, maj, dissent, P.std(axis=1), VD

    def path_agreement(self, X):
        '''How often the weighted vote and the stacked meta-learner disagree. The ns-3
        controller only blocks when its binary and 3-label outputs agree; that gate means
        nothing unless the two paths can actually differ, so the rate is reported.'''
        _, mv, _, _, _ = self.score(X, use_stack=False)
        _, ms, _, _, _ = self.score(X, use_stack=True)
        return float((mv != ms).mean()), int((mv != ms).sum())

    def predict_deploy(self, X):
        '''The path the inference engine actually runs. When the Warning channel does not use
        probability spread (tau = inf, as HAM's dissent-only head does) the calibrated models
        are never touched: hard votes alone give the decision, the dissent and the label. This
        is what the deployment latency in Table 11 measures.'''
        if HEADLINE_PATH.get(self.mode, DEFAULT_PATH) == 'vote' and not np.isfinite(self.tau):
            V = self._V(X); VD = V @ self.w
            maj = (VD >= self.maxvote).astype(int)
            dis = (V != maj[:, None]).astype(float) @ self.w
            warn = ((dis >= self.dissent_min) if np.isfinite(self.dissent_min)
                    else np.zeros(len(X), bool))
        else:
            _, maj, dis, spr, _ = self.score(X)
            warn = self._warn_mask(dis, spr, self.dissent_min, self.tau)
        return np.where(warn, 'Warning', np.where(maj == 1, 'Malicious', 'Benign')), maj

    @staticmethod
    def _warn_mask(dis, spr, d, s):
        '''The ONE definition of the Warning rule, used by fitting AND prediction.
        np.inf disables a channel. V8 shipped a 100%-warning head precisely because
        its fit-time and predict-time rules disagreed about a 0 threshold.'''
        m = np.zeros(len(dis), bool)
        if np.isfinite(d): m |= (dis >= d)
        if np.isfinite(s): m |= (spr >= s)
        return m

    def _fit_warning(self, Xv, yv):
        _, maj, dis, spr, _ = self.score(Xv)
        err = (maj != yv)
        d_grid = [np.inf] + sorted({float(v) for v in np.round(np.unique(dis),6) if v > 0})[:40]
        s_grid = [np.inf] + [float(v) for v in np.unique(np.quantile(spr, np.linspace(0.80,1.0,41)))]
        # Target the middle of the band, not its edge: this rate is measured on a
        # validation half and drifts on test, and a threshold chosen at the ceiling
        # trips the Cell 21 guard on drift alone. Widen only if nothing qualifies.
        # Cost is (errors let through) + lambda * (flows escalated). On this corpus that
        # landscape is flat -- many thresholds tie exactly, because cost lands on a 0.05 grid.
        # Ties are broken toward the SMALLEST warning rate: among equally good options, warn
        # as little as possible. Same principle as the MaxVoteIndex tie-break, and it makes
        # the choice deterministic instead of seed-dependent.
        best, bd, bs = (np.inf, np.inf), None, None
        # FIRST try dissent alone. Dissent is computed from hard votes, so a dissent-only head
        # lets predict_deploy skip the calibrated models entirely. Left to the joint search,
        # which channel wins is decided by the data and flips across seeds -- which would make
        # HAM's ~2x deployment speed-up a property of the seed rather than of the design.
        # Falls through to the joint search unchanged when dissent cannot reach the band.
        for ceil in (WARN_RATE_CEIL*0.6, WARN_RATE_CEIL):
            for d in d_grid:
                if not np.isfinite(d): continue
                w = self._warn_mask(dis, spr, d, np.inf); r = float(w.mean())
                if not (WARN_RATE_FLOOR <= r <= ceil): continue
                key = (err[~w].sum() + WARNING_PENALTY*w.sum(), r)
                if key < best: best, bd, bs = key, d, np.inf
            if bd is not None: break
        if bd is not None:
            print(f'  [{self.tag}] Warning channel: DISSENT only '
                  f'(dissent>={bd:.6g}) -- the deployment fast path applies')
        for ceil in ((WARN_RATE_CEIL*0.6, WARN_RATE_CEIL) if bd is None else ()):
            for d in d_grid:
                for s in s_grid:
                    w = self._warn_mask(dis, spr, d, s); r = float(w.mean())
                    if not (WARN_RATE_FLOOR <= r <= ceil): continue
                    key = (err[~w].sum() + WARNING_PENALTY*w.sum(), r)
                    if key < best: best, bd, bs = key, d, s
            if bd is not None: break
        if bd is None:
            # No threshold satisfied the cost search inside the band. Fall back on RATE alone:
            # take the grid point whose warning rate is closest to the band's midpoint. A fixed
            # quantile is not safe here -- when the spread distribution is heavily tied, the
            # 99th percentile can equal the maximum and produce a 0% rate, which then trips the
            # Cell 21 guard for a reason that has nothing to do with the model.
            mid = (WARN_RATE_FLOOR + WARN_RATE_CEIL) / 2
            cand = []
            for d in d_grid:
                for s in s_grid:
                    r = float(self._warn_mask(dis, spr, d, s).mean())
                    if r > 0: cand.append((abs(r - mid), r, d, s))
            if cand:
                _, r_sel, bd, bs = min(cand)
                print(f'  [{self.tag}] WARNING-HEAD FALLBACK: no in-band cost optimum; '
                      f'chose the rate closest to the band midpoint ({r_sel:.4%})')
            else:
                bd, bs = np.inf, np.inf
                print(f'  [{self.tag}] WARNING-HEAD DISABLED: no threshold produces any warning '
                      f'(members agree everywhere) -- reported as a 2-label mode')
        self.dissent_min, self.tau = bd, bs

    def predict3(self, X):
        p, maj, dis, spr, VD = self.score(X)
        warn = self._warn_mask(dis, spr, self.dissent_min, self.tau)
        lab = np.where(warn, 'Warning', np.where(maj==1, 'Malicious', 'Benign'))
        return lab, p, maj, warn, dis, spr, VD

    # ---- Per-member reporting: the contract the /logs/recent endpoint calls ----
    def individual_predictions(self, X):
        '''Each member's OWN hard vote, before the weighted vote, keyed by member name.

        Deliberately the same method name and the same return shape as V6's
        VMFCVDVoter.individual_predictions, so the deployed engine's per-base-model breakdown
        works against this object with no change to the engine. Diagnostic output only -- it
        never touches pred_binary or pred_3label.'''
        return {m.name: m.predict(X).astype(int) for m in self.members}

    def member_report(self, X, y=None):
        '''Per-member predictions ALWAYS; per-member metrics ONLY when ground truth is given.

        The engine calls this with y=None when the uploaded CSV has no Label column and with y
        set when it has one, so an unlabelled upload can never produce a metric computed against
        nothing. Depends on numpy and math alone -- no notebook-level helpers -- so models.py can
        carry this method verbatim.'''
        wmap = {m.name: float(v) for m, v in zip(self.members, self.w)}
        cost = {m.name: float(m.cost_us) for m in self.members}
        out = {}
        for name, p in self.individual_predictions(X).items():
            row = dict(mode=self.mode, member=name, shapley=wmap[name],
                       us_per_sample=cost[name], n=int(len(p)),
                       predicted_malicious=int((p == 1).sum()),
                       predicted_benign=int((p == 0).sum()), labelled=y is not None)
            if y is not None:
                yy = np.asarray(y).astype(int)
                tp = int(((p == 1) & (yy == 1)).sum()); tn = int(((p == 0) & (yy == 0)).sum())
                fp = int(((p == 1) & (yy == 0)).sum()); fn = int(((p == 0) & (yy == 1)).sum())
                den = math.sqrt(float(tp+fp) * float(tp+fn) * float(tn+fp) * float(tn+fn))
                prec = tp/(tp+fp) if tp+fp else 0.0
                rec  = tp/(tp+fn) if tp+fn else 0.0
                row.update(TP=tp, FN=fn, FP=fp, TN=tn,
                           MCC=((tp*tn - fp*fn)/den if den else 0.0),
                           Recall=rec, Precision=prec,
                           F1=(2*prec*rec/(prec+rec) if prec+rec else 0.0),
                           FPR=(fp/(fp+tn) if fp+tn else 0.0),
                           FNR=(fn/(fn+tp) if fn+tp else 0.0),
                           Accuracy=((tp+tn)/len(yy) if len(yy) else 0.0))
            out[name] = row
        return out


class DFDM:
    '''VMFCVD defensive mode: the FDM members over the FDM cluster, cheapest first,
    stopping at the first malicious vote. Benign requires unanimity.'''
    def __init__(self, fdm_ens): self.ens = fdm_ens
    def _ordered(self):
        return sorted(self.ens.members, key=lambda m: (m.cost_us if np.isfinite(m.cost_us) else 1e9))
    def predict3(self, X):
        '''Returns (labels, binary, warn, stage, stats). Stats are RETURNED, never stored on
        self: the latency probe re-runs this on a tiled batch, and stashing counters on the
        instance would leave the tiled numbers behind for the report to print.'''
        order = self._ordered()
        M, mi = Trident._packed(X, order)
        n = len(X); out = np.zeros(n, int); stage = np.full(n, -1)
        pend = np.ones(n, bool); exits = []
        for si, m in enumerate(order):
            if not pend.any(): exits.append(0); continue
            idx = np.flatnonzero(pend)
            hit = m.clf.predict(M[np.ix_(idx, mi[si])]).astype(bool)
            f = idx[hit]; out[f] = 1; stage[f] = si; pend[f] = False
            exits.append(int(hit.sum()))
        # DFDM is a TWO-label mode by design. Its only uncertainty signal would be a "late
        # catch" -- the cheapest member said benign and a costlier one disagreed -- which is a
        # different meaning of Warning from the other two modes' "the members disagreed", and
        # it fired on 1 flow in 25,462. More to the point, DFDM is selected above 5,000
        # flows/s, where there is no capacity to escalate anything to a human. Late catches
        # are still counted and reported as a diagnostic.
        warn = np.zeros(n, bool)
        lab = np.where(warn, 'Warning', np.where(out == 1, 'Malicious', 'Benign'))
        stats = dict(n=n, order=[m.name for m in order], exits=exits,
                     unanimous_benign=int(pend.sum()),
                     late_catches=int(((out == 1) & (stage > 0)).sum()))
        return lab, out, warn, stage, stats

    # DFDM votes over the SAME members as FDM, so its per-member breakdown IS FDM's. Exposed on
    # this object too, so the engine can call it on whichever object the flow rate resolved to.
    def individual_predictions(self, X): return self.ens.individual_predictions(X)
    def member_report(self, X, y=None):
        rep = self.ens.member_report(X, y)
        for r in rep.values(): r['mode'] = 'DFDM'
        return rep



# ═════════════════════════════════════════════════════════════════════════════
# Serving helpers carried over unchanged from the V6 engine
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


def compute_metrics(y_true, y_pred, mode_name='', verbose=True) -> dict:
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
    if verbose:
        tag = f'[{mode_name}] ' if mode_name else ''
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

