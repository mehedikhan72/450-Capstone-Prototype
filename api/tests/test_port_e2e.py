"""Drive the PORTED engine end to end, exactly as the Celery worker does."""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
API = os.path.dirname(_HERE)                      # .../api
ENG = os.path.join(API, "detection-engine")
sys.path.insert(0, ENG)
import models as _models
# exactly what inference_service._bind_main_module_classes() does
import __main__ as _m
for nm in ("LinRegImputer","Pre","Member","Trident","DFDM","DetailedResourceMonitor"):
    setattr(_m, nm, getattr(_models, nm))
import inference as inf, config as cfg

print('='*78); print('  1 · load_artifacts'); print('='*78)
art = inf.load_artifacts()
assert sorted(art)==['all_features','dfdm','fdm','fdm_features','ham','ham_features','pre','schema']
assert len(art['all_features'])==12, art['all_features']
assert len(art['fdm_features'])==2 and len(art['ham_features'])==11

print('\n'+'='*78); print('  2 · run_csv, single mode (FDM), labelled'); print('='*78)
res,out = inf.run_csv(csv_path=API+'/input/test_labelled.csv', flow_rate=1000,
                      label_col='Label', benign_label='Benign',
                      save_output=True, track_resources=False)
assert set(res)=={'FDM'}
r=res['FDM']
labs=set(pd.Series(r['predictions_3label']).unique())
assert labs <= {'Benign','Warning','Malicious'}, labs
assert r['predictions_3label'].dtype.kind in 'OU', r['predictions_3label'].dtype
assert {'pred_binary','pred_3label','mode'} <= set(out.columns)
assert 'src_ip' in out.columns and 'Label' in out.columns, 'pass-through lost'
assert out['pred_3label'].isin(['Benign','Warning','Malicious']).all()
assert r['metrics_binary'] and r['metrics_3label']
mr = r['model_report']
assert set(mr)== {'AdaBoost','Bagging','Boosting','RandomForest'}, set(mr)
one = mr['Bagging']
assert 'MCC' in one and one['labelled'] is True, one
print(f"  labels: {sorted(labs)}  | pass-through kept | per-member MCC present")
print(f"  member_report[Bagging]: mode={one['mode']} shapley={one['shapley']:.4f} MCC={one['MCC']:.4f}")

print('\n'+'='*78); print('  3 · unlabelled upload must yield NO metrics'); print('='*78)
res2,out2 = inf.run_csv(csv_path=API+'/input/test_unlabelled.csv', flow_rate=0,
                        label_col='Label', benign_label='Benign',
                        save_output=False, track_resources=False)
r2=res2['HAM']
assert 'metrics_binary' not in r2 and 'metrics_3label' not in r2
k=set(next(iter(r2['model_report'].values())))
banned={'MCC','Recall','Precision','F1','FPR','FNR','Accuracy','TP','FN','FP','TN'}
assert not (k & banned), k & banned
assert next(iter(r2['model_report'].values()))['labelled'] is False
print('  keys without a Label:', sorted(k)); print('  ✓ no metric leaked')

print('\n'+'='*78); print('  4 · all three modes (flow_rate omitted)'); print('='*78)
res3,out3 = inf.run_csv(csv_path=API+'/input/test_labelled.csv', flow_rate=None,
                        label_col='Label', benign_label='Benign',
                        save_output=False, track_resources=False)
assert set(res3)=={'HAM','FDM','DFDM'}, set(res3)
for m in ('HAM','FDM','DFDM'):
    assert f'pred_binary_{m}' in out3.columns and f'pred_3label_{m}' in out3.columns
dl=set(pd.Series(res3['DFDM']['predictions_3label']).unique())
assert dl <= {'Benign','Malicious'}, f'DFDM must be two-label, got {dl}'
print('  suffixed columns present for all three modes')
print('  DFDM label set:', sorted(dl), '(two-label by design ✓)')

print('\n'+'='*78); print('  5 · deployment path == full path (the Cell 21 guarantee)'); print('='*78)
X,_ = inf.preprocess(pd.read_csv(API+'/input/test_labelled.csv'), 'Label', 'Benign')
for tag,e in (('FDM',art['fdm']),('HAM',art['ham'])):
    lab_d, maj_d = e.predict_deploy(X)
    lab_f, _p, maj_f, *_ = e.predict3(X)
    ok = np.array_equal(lab_d,lab_f) and np.array_equal(maj_d,maj_f)
    print(f'  {tag}: labels+binary identical = {ok}')
    assert ok

print('\n'+'='*78); print('  6 · column tolerance'); print('='*78)
raw = pd.read_csv(API+'/input/test_labelled.csv')
for name, frame in (('12 cols exact', raw[art['all_features']]),
                    ('reversed order', raw[art['all_features'][::-1]]),
                    ('with pass-through', raw)):
    Xn,_ = inf.preprocess(frame.copy(), None, 'Benign')
    l,_b = art['ham'].predict_deploy(Xn)
    print(f'  {name:20s} -> OK, {Xn.shape[1]} cols, {len(l)} labels')
try:
    inf.preprocess(raw.drop(columns=['Packet Length Min']), None, 'Benign'); print('  !! missing column NOT caught')
except ValueError as ex:
    print('  missing column -> ValueError naming it:', str(ex).splitlines()[0][:60])

print('\n'+'='*78); print('  7 · assert_schema on an ordered frame'); print('='*78)
print('  HAM ordered :', inf.assert_schema(X[art['ham_features']], 'HAM'))
try:
    inf.assert_schema(X[art['ham_features'][::-1]], 'HAM'); print('  !! reversed accepted')
except ValueError: print('  reversed      : rejected ✓')

print('\n\nALL END-TO-END CHECKS PASSED')
