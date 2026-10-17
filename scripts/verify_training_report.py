"""Verify live report, cross-tab predictions, original CSV, and unchanged artifacts."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'service'))
from audit_engine import AuditEngine,ROOT,RATE
from flexible_engine import FlexibleEngine
from training_report import build_training_report
engine=AuditEngine();files=[ROOT/r['file'] for r in engine.report['models'].values()]+[ROOT/'data/train.csv',ROOT/'data/test.csv',ROOT/'models/feature_defaults.json']
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.relative_to(ROOT)):hashfile(p) for p in files};checks=[]
def check(name,ok):
    assert ok,name
    checks.append(name)
r=build_training_report(engine)
check('All 5 agree with original rounded CSV metrics',r['all_match_original_report'])
check('292 holdout disjoint from 1168 train',len(r['holdout_ids'])==292 and set(r['holdout_ids']).isdisjoint(set(engine.train_ids)))
tuning=json.loads((ROOT/'results/tuning_report.json').read_text())
expected=min(tuning['models'],key=lambda s:(tuning['models'][s]['selected']['cv_mae'],tuning['models'][s]['selected']['cv_std'],s))
check('Frozen train-only CV rule selects winner',r['winner']==expected)
check('CV winner in current report matches frozen selection',r['cv_winner']==expected)
import pandas as pd
old=pd.read_csv(ROOT/'results/baseline_original/model_results.csv').sort_values('CV_MAE_mean').iloc[0]
check('Preserved historical CV winner remains Gradient Boosting',old.Model=='Gradient Boosting')
flex=FlexibleEngine(engine)
app_report=flex.evaluate({'records':[{'source_id':i} for i in r['holdout_ids']],'models':[m['id'] for m in r['models']]})
y=engine.y_test.to_numpy()*RATE
for m,other in zip(r['models'],app_report['models']):
 p=np.array([row['prediction'] for row in m['rows']]);e=np.abs((p-y)/y)*100
 check(m['id']+' MAE/RMSE/R2 independently recomputed',np.isclose(m['metrics']['test']['mae'],mean_absolute_error(y,p)) and np.isclose(m['metrics']['test']['rmse'],np.sqrt(mean_squared_error(y,p))) and np.isclose(m['metrics']['test']['r2'],r2_score(y,p)))
 check(m['id']+' cross-tab predictions identical',np.array_equal(p,np.array([a['prediction'] for a in other['predictions']])))
 check(m['id']+' coverage recomputed from actual prices',all(m['coverage'][str(t)]==int(np.count_nonzero(e<=t)) for t in [5,10,20]))
 check(m['id']+' worst cases are true global worst',m['worst'][0]['percentage_error']==max(row['percentage_error'] for row in m['rows']))
 check(m['id']+' 292 actual prices match CSV',np.array_equal(y,np.array([row['actual'] for row in m['rows']])))
check('All 8 protected files unchanged',before=={str(p.relative_to(ROOT)):hashfile(p) for p in files})
check('Original online source evidence all 6 equivalent',all(f['equivalent_content'] for f in r['repository']['files']))
for f in json.loads((ROOT/'models/_backup_original/manifest.json').read_text())['files']:check('Immutable baseline '+f['backup'],hashfile(ROOT/f['backup'])==f['sha256'])
json.dumps(r,allow_nan=False)
(ROOT/'results/training_holdout_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'results/verification_training_report.json').write_text(json.dumps({'status':'PASS','passed':len(checks),'checks':checks,'protected_files':before},ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',len(checks),'training report checks')
