"""Prove portable IO never unpickles a Booster and preserves all predictions and original artifacts."""
import json,sys,zipfile,hashlib
from pathlib import Path
from unittest.mock import patch
import joblib,numpy as np,pandas as pd
from xgboost import Booster
from xgboost.core import XGBoostError
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'service'))
from model_io import load_pipeline,sha
from audit_engine import AuditEngine
checks=[]
def check(name,ok):
 assert ok,name
 checks.append(name)
file=ROOT/'models/final/xgboost.pkl'
train=pd.read_csv(ROOT/'data/train.csv').drop(columns=['SalePrice','Id']);test=pd.read_csv(ROOT/'data/test.csv').drop(columns='Id')
reference=ROOT/'results/xgboost_portable_reference.json'
try:
 original=joblib.load(file)
 expected=[original.predict(X) for X in [train,test]]
 payload={'source_pickle_sha256':sha(file),'train_data_sha256':sha(ROOT/'data/train.csv'),'test_data_sha256':sha(ROOT/'data/test.csv'),'train_predictions_usd':expected[0].tolist(),'test_predictions_usd':expected[1].tolist()}
 if reference.exists():check('Exported reference agrees with original pickle',json.loads(reference.read_text())==payload)
 else:reference.write_text(json.dumps(payload,indent=2),encoding='utf8')
except XGBoostError:
 # Windows may be unable to deserialize the original Linux snapshot. Compare to the
 # predictions exported from that snapshot on the source platform, not to new fits.
 payload=json.loads(reference.read_text())
 check('Reference corresponds to original pickle bytes',payload['source_pickle_sha256']==sha(file))
 check('Reference corresponds to original train/test data',payload['train_data_sha256']==sha(ROOT/'data/train.csv') and payload['test_data_sha256']==sha(ROOT/'data/test.csv'))
 expected=[np.asarray(payload['train_predictions_usd'],dtype=np.float32),np.asarray(payload['test_predictions_usd'],dtype=np.float32)]
with patch.object(Booster,'__setstate__',side_effect=RuntimeError('Simulated incompatible XGBoost memory snapshot')):
 loaded=load_pipeline(file)
 for label,X,p in zip(['All 1460 labeled records','All 1459 Kaggle test records'],[train,test],expected):
  actual=loaded.predict(X)
  check(label+': exact same predictions',np.array_equal(p,actual) and float(np.max(abs(p-actual)))==0)
 engine=AuditEngine();check('All five API models load with Booster unpickling blocked',len(engine.models)==5)
 for sid in engine.samples.Id.iloc[:2]:check('XGBoost UBJ trace '+str(sid),engine.predict('xgboost',int(sid))['trace']['verified'])
manifest=json.loads((ROOT/'models/final/manifest.json').read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
check('Manifest is strict JSON with no NaN/Infinity',manifest['models']['xgboost']['portable']['format']=='xgboost.save_model UBJSON')
# Validate portable companions against the source weight hash and final manifest.
info=manifest['models']['xgboost']['portable'];check('Original XGBoost pickle still intact',sha(file)==info['source_pickle_sha256'])
for f,h in info['files'].items():check('Portable SHA256 '+f,sha(file.parent/f)==h)
backup=json.loads((ROOT/'models/_backup_original/manifest.json').read_text())
for f,h in backup['data_hashes'].items():check('Unchanged data '+f,sha(ROOT/f)==h)
for f in backup['files']:check('Unchanged original backup '+f['backup'],sha(ROOT/f['backup'])==f['sha256'])
result={'status':'PASS','passed':len(checks),'checks':checks,'platform_tested':sys.platform,'windows_native_execution':'Not available in this environment; simulate incompatible Booster pickle and use official portable model IO.','maximum_prediction_difference_usd':0.0,'retrained':False}
(ROOT/'results/verification_portable_xgboost.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('PASS',len(checks),'portable checks; max prediction difference = 0 USD; Booster pickle blocked; no retraining.')
