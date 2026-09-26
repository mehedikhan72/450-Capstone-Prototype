import glob
import hashlib
import io
import os
import platform
import shutil
import sys
import time
import traceback
import uuid
import zipfile

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

import models
from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.feedback import Annotation, Prediction
from app.models.lifecycle import DatasetSnapshot, DeploymentState, ModelVersion, TrainingRun


REGISTRY_DIR = os.environ.get("MODEL_REGISTRY_DIR", "/registry")
BASE_DATASET_PATH = os.environ.get("BASE_DATASET_PATH", "/data/DDoS_Dataset.zip")
FEEDBACK_WEIGHT = int(os.environ.get("FEEDBACK_WEIGHT", "5"))
LATENCY_BUDGET_US = {"FDM": 20, "HAM": 75}
IN_SCOPE = ["Syn-training", "Syn-testing", "UDP-training", "UDP-testing",
            "UDPLag-training", "UDPLag-testing"]
CLASS_MAP = {"Benign": "Benign", "Syn": "SYN", "UDP": "UDPflood", "DrDoS_UDP": "UDPflood"}


def train(run_id: str):
    session = SessionLocal()
    run = session.get(TrainingRun, run_id)
    snapshot = session.get(DatasetSnapshot, run.snapshot_id) if run else None
    try:
        if not run or not snapshot: raise ValueError("training run not found")
        run.status = "building"; snapshot.status = "building"; session.commit()
        parent_path = _parent_path(session, run)
        parent = _load_bundle(parent_path)
        schema = parent["schema"]
        features = list(schema["deploy_features"])
        train_df, val_df, test_df = _base_splits(snapshot.base_sha256, features)
        feedback = _feedback(session, snapshot.annotation_cutoff, features)
        snapshot.feedback_rows = len(feedback)
        snap_dir = os.path.join(REGISTRY_DIR, "snapshots", snapshot.id)
        os.makedirs(snap_dir, exist_ok=True)
        feedback.to_parquet(os.path.join(snap_dir, "feedback.parquet"), index=False)
        snapshot.path = snap_dir; snapshot.status = "ready"; run.status = "training"; session.commit()

        if len(feedback):
            train_df = pd.concat([train_df] + [feedback] * max(1, FEEDBACK_WEIGHT), ignore_index=True)
        pre = models.Pre().fit(train_df, features)
        Xtr, Xva, Xte = (pre.transform(frame) for frame in (train_df, val_df, test_df))
        ytr, yva, yte = (_labels(frame) for frame in (train_df, val_df, test_df))
        tuned = {m.name: dict(m.params) for m in parent["ham"].members}
        fdm = models.Trident(schema["fdm_members"], schema["fdm_features"], "FDM", tuned, mode="FDM").fit(Xtr, ytr)
        ham = models.Trident(schema["ham_members"], schema["ham_features"], "HAM", tuned, mode="HAM").fit(Xtr, ytr)
        groups = val_df["_grp"].to_numpy()
        fdm.calibrate(Xva, yva, groups); ham.calibrate(Xva, yva, groups)
        for candidate, previous in ((fdm, parent["fdm"]), (ham, parent["ham"])):
            costs = {m.name: m.cost_us for m in previous.members}
            for member in candidate.members: member.cost_us = costs.get(member.name, np.inf)
        dfdm = models.DFDM(fdm)

        run.status = "validating"; session.commit()
        report = {mode: _compare(mode, parent[mode.lower()], candidate, parent["pre_deploy"], pre,
                                 test_df, yte) for mode, candidate in (("FDM", fdm), ("HAM", ham))}
        passed = all(row["passed"] for row in report.values())
        report["passed"] = passed
        run.metrics = report
        if not passed:
            run.status = "rejected"; session.commit(); return

        model_id = f"trident-{uuid.uuid4().hex[:12]}"
        new_schema = dict(schema, version=model_id, parent_version=schema.get("version"),
                          dataset_snapshot=snapshot.id, feedback_rows=len(feedback),
                          env={"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "sklearn": sklearn.__version__,
                               "lightgbm": lightgbm.__version__, "joblib": joblib.__version__})
        model_dir = os.path.join(REGISTRY_DIR, "models", model_id)
        os.makedirs(model_dir, exist_ok=True)
        artifact = os.path.join(model_dir, "model.joblib")
        joblib.dump(dict(pre=pre, pre_deploy=pre, fdm=fdm, ham=ham, dfdm=dfdm,
                         schema=new_schema), artifact, compress=3)
        digest = _sha256(artifact)
        slot_dir = os.path.join(REGISTRY_DIR, "slots", run.target_slot)
        os.makedirs(slot_dir, exist_ok=True)
        temp = os.path.join(slot_dir, "model.joblib.tmp")
        shutil.copy2(artifact, temp); os.replace(temp, os.path.join(slot_dir, "model.joblib"))
        model = ModelVersion(id=model_id, parent_id=run.parent_model_id,
                             training_run_id=run.id, slot=run.target_slot, status="reloading",
                             artifact_path=artifact, artifact_sha256=digest, metrics=report)
        session.add(model); run.status = "reloading"; session.commit()
        celery_app.send_task("reload_model", args=[model_id, os.path.join(slot_dir, "model.joblib")],
                             queue=f"inference_{run.target_slot}")
    except Exception as error:
        if run:
            run.status = "failed"; run.error_message = f"{error}\n{traceback.format_exc()[-4000:]}"
        if snapshot and snapshot.status != "ready":
            snapshot.status = "failed"; snapshot.error_message = str(error)
        session.commit()
    finally:
        session.close()


def _parent_path(session, run):
    if run.parent_model_id:
        model = session.get(ModelVersion, run.parent_model_id)
        if model and os.path.isfile(model.artifact_path): return model.artifact_path
    configured = os.environ.get("MODEL_BUNDLE_PATH")
    if configured and os.path.isfile(configured): return configured
    found = glob.glob("/app/detection-engine/weights/*.joblib")
    if len(found) != 1: raise ValueError("exactly one bootstrap model bundle is required")
    return found[0]


def _load_bundle(path):
    main = sys.modules["__main__"]
    for name in ("LinRegImputer", "Pre", "Member", "Trident", "DFDM"):
        setattr(main, name, getattr(models, name))
    return joblib.load(path)


def _base_splits(digest, deploy_features):
    cache = os.path.join(REGISTRY_DIR, "base", digest)
    paths = [os.path.join(cache, f"{name}.parquet") for name in ("train", "val", "test")]
    if all(os.path.isfile(path) for path in paths):
        return tuple(pd.read_parquet(path) for path in paths)
    os.makedirs(cache, exist_ok=True)
    with zipfile.ZipFile(BASE_DATASET_PATH) as archive:
        parts = []
        for stem in IN_SCOPE:
            parts.append(pd.read_parquet(io.BytesIO(archive.read(f"{stem}.parquet"))))
        for name in archive.namelist():
            if name[:-8] in IN_SCOPE or not name.endswith(".parquet"): continue
            frame = pd.read_parquet(io.BytesIO(archive.read(name)))
            benign = frame[frame["Label"].astype(str).str.strip() == "Benign"]
            if len(benign): parts.append(benign)
    raw = pd.concat(parts, ignore_index=True)
    raw.columns = [column.strip() for column in raw.columns]
    raw["Label"] = raw["Label"].astype(str).str.strip()
    raw["cls"] = raw["Label"].map(CLASS_MAP)
    raw = raw[raw["cls"].notna()].copy()
    for column in ("Fwd Header Length", "Bwd Header Length", "Fwd Seg Size Min"):
        if column in raw: raw.loc[raw[column] < 0, column] = np.nan
    numeric = [c for c in raw if c not in ("Label", "cls") and np.issubdtype(raw[c].dtype, np.number)]
    drop = ([c for c in numeric if raw[c].isna().mean() > .4]
            + [c for c in numeric if raw[c].nunique(dropna=False) <= 1]
            + [c for c in raw if c.startswith("Subflow")]
            + [c for c in ("Flow Bytes/s", "Flow Packets/s") if c in raw])
    raw = raw.drop(columns=list(dict.fromkeys(drop)))
    features = [c for c in raw if c not in ("Label", "cls") and np.issubdtype(raw[c].dtype, np.number)]
    conflict = raw.groupby(features, observed=True, dropna=False)["cls"].transform("nunique") > 1
    raw = raw[~conflict].drop_duplicates(subset=features).reset_index(drop=True)
    raw[features] = raw[features].replace([np.inf, -np.inf], np.nan)
    raw["_grp"] = _groups(raw, features)
    splits = _split(raw)
    keep = deploy_features + ["Label", "cls", "_grp"]
    result = tuple(frame[keep].copy() for frame in splits)
    for frame, path in zip(result, paths): frame.to_parquet(path, index=False)
    return result


def _groups(frame, features):
    values = frame[features].fillna(frame[features].median(numeric_only=True)).to_numpy("float64")
    scaled = np.round(StandardScaler().fit_transform(values), 2)
    view = np.ascontiguousarray(scaled).view([("", scaled.dtype)] * scaled.shape[1]).ravel()
    return np.unique(view, return_inverse=True)[1]


def _split(frame):
    key = frame["cls"].astype(str) + "|" + frame["Protocol"].astype(str)
    counts = key.value_counts(); keep = ~key.isin(set(counts[counts < 10].index))
    frame, key = frame[keep].copy(), key[keep]
    groups = (pd.DataFrame({"grp": frame["_grp"].values, "key": key.values})
              .groupby("grp")["key"].agg(size="size", key=lambda s: s.mode().iat[0]).reset_index())
    groups = groups.iloc[np.random.RandomState(42).permutation(len(groups))]
    assignment = {}
    for _, subset in groups.groupby("key"):
        subset = subset.sort_values("size", ascending=False, kind="mergesort")
        quota = dict(zip(("train", "val", "test"), np.array((.7, .15, .15)) * subset["size"].sum()))
        filled = {name: 0 for name in quota}
        for group, size in zip(subset["grp"], subset["size"]):
            name = max(quota, key=lambda item: quota[item] - filled[item])
            assignment[group] = name; filled[name] += size
    part = frame["_grp"].map(assignment)
    return tuple(frame[part == name].copy() for name in ("train", "val", "test"))


def _feedback(session, cutoff, features):
    latest = {}
    for annotation in session.scalars(select(Annotation).where(Annotation.created_at <= cutoff)
                                      .order_by(Annotation.created_at, Annotation.id)):
        latest[annotation.prediction_id] = annotation
    if not latest: return pd.DataFrame(columns=features + ["Label", "cls", "_grp"])
    rows = []
    for prediction in session.scalars(select(Prediction).where(Prediction.id.in_(latest))):
        data = prediction.row_data
        if all(feature in data and data[feature] is not None for feature in features):
            label = latest[prediction.id].label
            rows.append({**{f: data[f] for f in features}, "Label": label,
                         "cls": label, "_grp": int(hashlib.sha256(
                             prediction.source_row_key.encode()).hexdigest()[:8], 16)})
    return pd.DataFrame(rows, columns=features + ["Label", "cls", "_grp"])


def _labels(frame): return (frame["Label"].astype(str) != "Benign").astype(int).to_numpy()


def _compare(mode, old, new, old_pre, new_pre, test, labels):
    old_x, new_x = old_pre.transform(test), new_pre.transform(test)
    old_pred = old.predict_deploy(old_x)[1]
    new_pred = new.predict_deploy(new_x)[1]
    def score(pred):
        tn, fp, fn, tp = metrics.confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        return {"mcc": metrics.matthews_corrcoef(labels, pred), "recall": tp/max(1,tp+fn),
                "fpr": fp/max(1,fp+tn)}
    before, after = score(old_pred), score(new_pred)
    def latency(model, values):
        sample = values[:min(20000, len(values))]
        started = time.perf_counter(); model.predict_deploy(sample)
        return (time.perf_counter() - started) * 1e6 / max(1, len(sample))
    before["latency_us"] = latency(old, old_x)
    after["latency_us"] = latency(new, new_x)
    budget = LATENCY_BUDGET_US[mode]
    passed = (after["mcc"] >= before["mcc"] - .005 and
              after["recall"] >= before["recall"] - .005 and
              after["fpr"] <= max(.01, before["fpr"] + .002) and
              after["latency_us"] <= budget)
    return {"current": before, "candidate": after, "passed": passed}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
