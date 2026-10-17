"""Regression checks for flexible, labeled and unlabeled evaluation; no fit calls."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'service'))
from audit_engine import AuditEngine,MODEL_IDS,RATE,sha
from flexible_engine import FlexibleEngine
engine=AuditEngine();flex=FlexibleEngine(engine);checks=[]
def check(name,condition):
    checks.append({'check':name,'passed':bool(condition)})
    if not condition:raise AssertionError(name)
def invalid(name,payload):
    try:flex.evaluate(payload)
    except (ValueError,TypeError):check(name,True)
    else:check(name,False)
record=flex.lookup(1000)
known=flex.evaluate({'records':[{'source_id':1000}],'models':MODEL_IDS,'explain':True})
check('Lookup beyond fixed demo allowed',1000 not in set(engine.samples.Id))
check('Known full row verifies original label',known['rows'][0]['truth']=='original_csv')
for m in known['models']:
    check(m['id']+' direct pipeline matches dynamic result',np.isclose(m['predictions'][0]['prediction'],float(engine.models[m['id']].predict(engine.df.loc[engine.df.Id==1000].drop(columns='SalePrice'))[0])*RATE))
    check(m['id']+' actual decomposition',m['trace']['verified'])
    check(m['id']+' stage final matches predict',np.isclose(m['stage_evaluation'][-1]['prediction'],m['predictions'][0]['prediction'],atol=1,rtol=1e-9))
    check(m['id']+' one-row R2 missing rather than invented',m['metrics']['r2'] is None)
new={'features':{'GrLivArea':1900,'OverallQual':7,'LotArea':9000,'YearBuilt':2001}}
unlabeled=flex.evaluate({'records':[new],'models':MODEL_IDS})
check('Unknown house no fabricated actual price',unlabeled['rows'][0]['actual'] is None)
check('Unlabeled predictions have no fake metrics or winner',all(m['metrics'] is None for m in unlabeled['models']) and unlabeled['best_by_mae'] is None)
check('Default feature filling disclosed',len(unlabeled['rows'][0]['defaulted_columns'])==len(flex.columns)-len(new['features']))
changed=flex.evaluate({'records':[{'source_id':1000,'features':{'OverallQual':1}}],'models':['xgboost']})
check('Changed source features lose original actual',changed['rows'][0]['actual'] is None)
manual=flex.evaluate({'records':[{**new,'actual_vnd':5000000000}],'models':MODEL_IDS,'explain':True})
check('Manual label provenance is not verified',manual['rows'][0]['truth']=='user_supplied')
for m in manual['models']:
    error=m['predictions'][0]['prediction']-5000000000
    check(m['id']+' current MSE is actual squared residual',np.isclose(m['metrics']['mse'],error*error))
    check(m['id']+' same-label baseline MSE correct',np.isclose(manual['baseline']['mse'],(manual['baseline_prediction']-5000000000)**2))
    check(m['id']+' stage MSE from actual label',np.isclose(m['stage_evaluation'][-1]['squared_error'],error*error,atol=1,rtol=1e-9))
frame=pd.read_csv(ROOT/'public/examples/new_labeled_12_VND.csv')
records=[{'id':str(int(row.Id)),'features':flex.clean(row.drop('ActualPriceVND').to_dict()),'actual_vnd':float(row.ActualPriceVND)} for _,row in frame.iterrows()]
batch=flex.evaluate({'records':records,'models':MODEL_IDS})
check('New labeled CSV has 12 rows outside old 30',len(batch['rows'])==12 and not set(frame.Id)&set(engine.samples.Id))
check('All original labels of new CSV verified',all(r['truth']=='original_csv' and r['membership']=='holdout' for r in batch['rows']))
y=frame.ActualPriceVND.to_numpy(float)
for m in batch['models']:
    p=np.array([r['prediction'] for r in m['predictions']])
    check(m['id']+' new dataset MAE sklearn',np.isclose(m['metrics']['mae'],mean_absolute_error(y,p)))
    check(m['id']+' new dataset MSE sklearn',np.isclose(m['metrics']['mse'],mean_squared_error(y,p)))
    check(m['id']+' new dataset R2 sklearn',np.isclose(m['metrics']['r2'],r2_score(y,p)))
    check(m['id']+' new metrics not reused from demo30',not np.isclose(m['metrics']['mae'],engine.report['models'][m['id']]['metrics']['demo']['mae']*RATE))
partial=flex.evaluate({'records':records[:2]+[new],'models':['forest']})
check('Mixed labels use only labeled rows',partial['labeled_count']==2 and partial['models'][0]['metrics']['n']==2 and partial['models'][0]['predictions'][2]['squared_error'] is None)
constant=flex.evaluate({'records':[{**new,'actual_vnd':5e9},{**new,'actual_vnd':5e9}],'models':['linear']})
check('Constant labels do not fabricate R2',constant['models'][0]['metrics']['r2'] is None)
usd=flex.evaluate({'records':[{**new,'actual':200000}],'models':['tree'],'target_currency':'USD'})
check('SalePrice USD converts once',usd['rows'][0]['actual']==5e9)
vnd=flex.evaluate({'records':[{**new,'actual_vnd':5e9}],'models':['tree'],'target_currency':'USD'})
check('Explicit actual_vnd never double converted',vnd['rows'][0]['actual']==5e9)
kaggle=flex.dataset('kaggle-test');check('All 1459 Kaggle test inputs accessible',len(kaggle['records'])==1459)
unknown=flex.evaluate({'records':kaggle['records'][:4],'models':MODEL_IDS})
check('Kaggle test never receives original train labels',unknown['labeled_count']==0 and all(r['actual'] is None for r in unknown['rows']))
for name,payload in [('Reject missing records',{}),('Reject submission-only Id/price',{'records':[{'features':{'Id':2000},'actual_vnd':5e9}]}),('Reject overflow label',{'records':[{**new,'actual_vnd':1e308}]}),('Reject negative label',{'records':[{**new,'actual_vnd':-1}]}),('Reject unknown field',{'records':[{'features':{'bad':1}}]}),('Reject invalid numeric',{'records':[{'features':{'GrLivArea':'abc'}}]}),('Reject invalid label units',{'records':[new],'target_currency':'EUR'})]:invalid(name,payload)
for file,digest in engine.report['source_files'].items():check('Preserved '+file,sha(ROOT/file)==digest)
json.dumps(batch,allow_nan=False);check('JSON contains no NaN/Infinity',True)
result={'status':'PASS','passed':len(checks),'failed':0,'checks':checks,'scope':'Dynamic actual labels, new records, model decomposition, metrics and protected artifacts; not external market price verification.'}
(ROOT/'results/verification_dynamic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',len(checks),'dynamic numerical checks')
