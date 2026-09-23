# Port smoke test — no Kaggle artefact needed

These two scripts prove the v13 port works end to end **without** the real
`trident_v13.joblib`, which is gitignored and lives only on the training host.

```bash
python tests/make_synthetic_bundle.py   # fits a tiny TRIDENT on synthetic flows
python tests/test_port_e2e.py           # drives inference.py exactly as the worker does
```

`make_synthetic_bundle.py` writes `detection-engine/weights/trident_v13.joblib`.
**Delete it before dropping in the real model** — `_find_bundle()` refuses to start
if it finds two `.joblib` files, which is the safe failure, but a synthetic model
sitting in `weights/` is still a trap.

What the test covers: bundle loading and the startup audit trail, the version-mismatch
warning, `preprocess` through `pre_deploy`, all three modes, string labels end to end,
DFDM staying two-label, `predict_deploy` == `predict3`, the 12/reordered/extra-column
tolerance, the missing-column error, and `assert_schema`'s negative control.

What it does **not** cover: LightGBM unpickling. The `Boosting` member is an
`LGBMClassifier` in the real bundle; if lightgbm is not installed when this runs, the
synthetic bundle falls back to `GradientBoostingClassifier`. Install `lightgbm==4.6.0`
(now in `requirements.txt`) before trusting the real artefact.
