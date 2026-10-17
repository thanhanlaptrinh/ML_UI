"""Independent checks against saved pipelines, original labels and sklearn metrics."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'service'))
from audit_engine import AuditEngine, RATE, CATALOG, sha
engine=AuditEngine();checks=[]
def check(name,condition):
    checks.append({'check':name,'passed':bool(condition)})
    if not condition:raise AssertionError(name)
for file,digest in engine.report['source_files'].items():check('Protected SHA-256 '+file,sha(ROOT/file)==digest)
check('Demo IDs absent from reconstructed training split',not set(engine.samples.Id)&set(engine.train_ids))
check('Exactly 30 distinct source IDs',len(engine.samples.Id.unique())==30)
old=pd.read_csv(ROOT/'results/model_results.csv',encoding='utf-8-sig').set_index('Model')
for slug,name,_ in CATALOG:
    X=engine.samples.drop(columns=['SalePrice','Id']);y=engine.samples.SalePrice.to_numpy(float)*RATE
    p=engine.models[slug].predict(X).astype(float)*RATE
    calc=engine.calculations(slug);m=calc['metrics']
    check(f'{slug}: original CSV SalePrice matches all 30 labels',all(r['actual']==float(engine.df.loc[engine.df.Id==r['id'],'SalePrice'].iloc[0])*RATE for r in calc['rows']))
    for key,value in [('mae',mean_absolute_error(y,p)),('mse',mean_squared_error(y,p)),('rmse',np.sqrt(mean_squared_error(y,p))),('r2',r2_score(y,p))]:
        check(f'{slug}: {key} independent recomputation',np.isclose(m[key],value,rtol=1e-9,atol=1e-6))
    check(f'{slug}: VND scaling preserves R2',np.isclose(m['r2'],r2_score(y/RATE,p/RATE),rtol=1e-10))
    check(f'{slug}: 292-row holdout MAE matches generated final CSV',abs(engine.report['models'][slug]['metrics']['test']['mae']-old.loc[name,'MAE'])<.011)
    for sample_id in engine.samples.Id.iloc[:2]:
        r=engine.predict(slug,int(sample_id));check(f'{slug}: model decomposition ID={int(sample_id)}',r['trace']['verified'])
changed=engine.predict('xgboost',int(engine.samples.Id.iloc[0]),{'OverallQual':1})
check('Edited hypothetical home has no claimed ground truth',changed['actual'] is None)
report={'status':'PASS','passed':len(checks),'failed':0,'rate':RATE,'checks':checks,
        'scope':'Arithmetic and artifact integrity verified; this does not assert exact market prices or external transaction verification.'}
(ROOT/'results/verification_vnd.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',len(checks),'checks')
for slug,name,_ in CATALOG:print(name,engine.report['models'][slug]['assessment']['label'])
