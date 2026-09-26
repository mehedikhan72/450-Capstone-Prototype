"""Build a synthetic TRIDENT v13 bundle using the PORTED models.py, so the engine
can be driven end to end without the real Kaggle artefact."""
import sys, os, io, json, numpy as np, pandas as pd, joblib, platform, sklearn

_HERE = os.path.dirname(os.path.abspath(__file__))
API = os.path.dirname(_HERE)                      # .../api
ENG = os.path.join(API, "detection-engine")
sys.path.insert(0, ENG)
import models as M

FDM_F = ['Packet Length Min', 'ACK Flag Count']
HAM_F = ['Packet Length Min', 'Fwd Packets/s', 'Down/Up Ratio', 'URG Flag Count',
         'Fwd Packets Length Total', 'Init Fwd Win Bytes', 'Init Bwd Win Bytes',
         'Total Fwd Packets', 'Fwd Packet Length Max', 'Bwd Packet Length Max',
         'Avg Packet Size']
DEPLOY = ['Total Fwd Packets','Fwd Packets Length Total','Fwd Packet Length Max',
          'Bwd Packet Length Max','Packet Length Min','Avg Packet Size','Fwd Packets/s',
          'Down/Up Ratio','ACK Flag Count','URG Flag Count','Init Fwd Win Bytes',
          'Init Bwd Win Bytes']
assert set(DEPLOY) == set(FDM_F) | set(HAM_F), 'deploy union mismatch'

rng = np.random.RandomState(0); n = 4000
attack = rng.rand(n) < 0.45
df = pd.DataFrame({
    'Total Fwd Packets'       : np.where(attack, rng.poisson(3,n), rng.poisson(40,n)).astype(float),
    'Fwd Packets Length Total': np.where(attack, rng.lognormal(6,.4,n), rng.lognormal(8,1,n)),
    'Fwd Packet Length Max'   : np.where(attack, 1472.0, rng.lognormal(6,1,n)),
    'Bwd Packet Length Max'   : np.where(attack, 0.0, rng.lognormal(6,1,n)),
    'Packet Length Min'       : np.where(attack, 1472.0, rng.lognormal(3,1,n)),
    'Avg Packet Size'         : np.where(attack, 1480.0, rng.lognormal(5,1,n)),
    'Fwd Packets/s'           : np.where(attack, rng.lognormal(11,.5,n), rng.lognormal(5,2,n)),
    'Down/Up Ratio'           : np.where(attack, 0.0, rng.poisson(1,n)).astype(float),
    'ACK Flag Count'          : np.where(attack, 0.0, (rng.rand(n)<.8).astype(float)),
    'URG Flag Count'          : (rng.rand(n)<.05).astype(float),
    'Init Fwd Win Bytes'      : np.where(attack, -1.0, 65535.0),
    'Init Bwd Win Bytes'      : np.where(attack, -1.0, rng.choice([-1,229,65535],n).astype(float)),
})[DEPLOY]
y = attack.astype(int)
grp = np.arange(n)
tr, va = slice(0,2600), slice(2600,4000)

M.N_ESTIMATORS = 12
pre_deploy = M.Pre().fit(df.iloc[tr], DEPLOY)
Xtr, Xva = pre_deploy.transform(df.iloc[tr]), pre_deploy.transform(df.iloc[va])
fdm = M.Trident(['AdaBoost','Bagging','Boosting','RandomForest'], FDM_F, 'FDM', {}, mode='FDM')\
       .fit(Xtr, y[tr]).calibrate(Xva, y[va], grp[va])
ham = M.Trident(['AdaBoost','Bagging','Boosting','RandomForest','ExtraTrees'], HAM_F, 'HAM', {}, mode='HAM')\
       .fit(Xtr, y[tr]).calibrate(Xva, y[va], grp[va])
dfdm = M.DFDM(fdm)
for m in fdm.members: m.cost_us = 1.0

SCHEMA = dict(version='TRIDENT v13', build='v13.1 · SYNTHETIC test bundle',
              env=dict(python=platform.python_version(), numpy=np.__version__,
                       pandas=pd.__version__, scikit_learn=sklearn.__version__,
                       lightgbm='4.6.0'),
              random_state=42, all_features=DEPLOY, deploy_features=DEPLOY,
              fdm_features=FDM_F, ham_features=HAM_F,
              fdm_members=[m.name for m in fdm.members],
              ham_members=[m.name for m in ham.members],
              maxvote=dict(FDM=float(fdm.maxvote), HAM=float(ham.maxvote)),
              flow_thresholds=dict(high=1000, extreme=5000))

import __main__ as _m
for nm in ('LinRegImputer','Pre','Member','Trident','DFDM'): setattr(_m, nm, getattr(M, nm))
out = os.path.join(ENG, 'weights', 'trident_v13.joblib')
os.makedirs(os.path.dirname(out), exist_ok=True)
joblib.dump(dict(pre=pre_deploy, pre_deploy=pre_deploy, fdm=fdm, ham=ham, dfdm=dfdm,
                 schema=SCHEMA), out, compress=3)
print('bundle written:', out, f'{os.path.getsize(out)/1024:.0f} KB')

# test CSVs: 12 cols + Label, and one with pass-through + no Label
te = df.iloc[va].reset_index(drop=True).copy()
te['Label'] = np.where(y[va]==1, 'UDPflood', 'Benign')
te['src_ip'] = ['10.0.0.%d' % (i % 250) for i in range(len(te))]
te.to_csv(os.path.join(ENG,'..','input','test_labelled.csv'), index=False)
te.drop(columns=['Label']).to_csv(os.path.join(ENG,'..','input','test_unlabelled.csv'), index=False)
print('test CSVs written; attack share %.3f' % y[va].mean())
