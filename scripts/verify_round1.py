"""Non-regression checks specific to round 1: immutable baseline, selection isolation, Id invariant."""
import sys,json,subprocess
from pathlib import Path
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'service'),str(ROOT/'notebooks_or_src_py')]
from audit_engine import AuditEngine,RATE,sha
from flexible_engine import FlexibleEngine
checks=[]
def check(name,ok):
 assert ok,name
 checks.append(name)
e=AuditEngine();f=FlexibleEngine(e);manifest=json.loads((ROOT/'models/final/manifest.json').read_text());selection=json.loads((ROOT/'results/tuning_selection.json').read_text());report=json.loads((ROOT/'results/tuning_report.json').read_text());backup=json.loads((ROOT/'models/_backup_original/manifest.json').read_text())
check('Exactly 79 feature names and no Id',len(f.columns)==79 and 'Id' not in f.columns and 'Id' not in f.defaults)
check('Defaults/schema exclude Id',len(f.schema()['columns'])==79 and 'Id' not in f.schema()['defaults'])
for path,h in backup['data_hashes'].items():check('Original data unchanged '+path,sha(ROOT/path)==h)
for info in backup['files']:check('Immutable backup '+info['backup'],sha(ROOT/info['backup'])==info['sha256'])
check('Training reads selected parameters from tuning_report',all(report['models'][s]['selected']==selection['models'][s]['selected'] for s in selection['models']))
check('Frozen train-only selection hash',sha(ROOT/'results/tuning_selection.json')==manifest['selection_sha256']==report['selection_sha256'])
check('Selection never contains holdout metrics',all('holdout' not in json.dumps(m).lower() for m in selection['models'].values()))
check('Selection row identities exactly train',selection['train_ids']==e.train_ids and set(selection['train_ids']).isdisjoint(e.test_ids))
check('All five official compressed pipelines saved',len(manifest['models'])==5 and set(manifest['models'])==set(e.models))
for slug,pipe in e.models.items():
 info=manifest['models'][slug];check(slug+' hash and file size valid',sha(ROOT/info['file'])==info['sha256'] and (ROOT/info['file']).stat().st_size==info['bytes'])
 check(slug+' pipeline sees 79 inputs',len(pipe.feature_names_in_)==79 and 'Id' not in pipe.feature_names_in_)
 frame=e.X_test.iloc[:200].copy();expected=pipe.predict(frame)
 for value in [1,400,800,1200,1460,999999]:
  changed=frame.assign(Id=value);actual=pipe.predict(changed)
  check(slug+f' Id={value}: max prediction delta exactly zero',np.array_equal(expected,actual) and float(np.max(abs(expected-actual)))==0)
 # Same API inputs including arbitrary Id metadata must also produce the exact same number.
 values=[]
 for value in [1,400,800,1200,1460]:
  fields=f.lookup(1000)['features'];fields['Id']=value
  r=f.evaluate({'records':[{'features':fields}],'models':[slug]});values.append(r['models'][0]['predictions'][0]['prediction'])
 check(slug+' Id invariant through FlexibleEngine',max(values)-min(values)==0)
 m=selection['models'][slug];check(slug+' final CV no worse than no-Id original-parameter control',m['selected']['cv_mae']<=m['original_without_Id_cv']['mae']+1e-8)
 if slug in ['forest','boosting','xgboost']:
  check(slug+' 40 randomized trials for raw and log',all(s['n_iter']==40 for s in m['searches'][:2]))
  for search in m['searches'][:2]:check(slug+' refit and neg-MAE scoring recorded '+search['variant'],search['refit'] is True and search['best_score_']<0)
 if slug=='tree':check('Decision Tree full grid 160 configurations',m['searches'][0]['n_iter']==160)
check('Winner selected only by CV mean/std',manifest['winner']==min(selection['models'],key=lambda s:(selection['models'][s]['selected']['cv_mae'],selection['models'][s]['selected']['cv_std'],s)))
check('Best-model copy matches CV winner bytes',sha(ROOT/'models/best_model.pkl')==manifest['models'][manifest['winner']]['sha256'])
# Refusal must occur before any fitting or writing; verify all hashes stay unchanged.
protected={str(p):sha(p) for p in [ROOT/'models/best_model.pkl',ROOT/'models/final/manifest.json',ROOT/'results/tuning_report.json']}
proc=subprocess.run([sys.executable,str(ROOT/'notebooks_or_src_py/train_all.py')],capture_output=True,text=True)
check('Without --force train_all refuses with explicit error',proc.returncode!=0 and 'use --force' in proc.stderr)
check('Refusal preserves all protected bytes',all(sha(Path(p))==h for p,h in protected.items()))
json.dumps(report,allow_nan=False)
(ROOT/'results/verification_round1.json').write_text(json.dumps({'status':'PASS','passed':len(checks),'checks':checks},ensure_ascii=False,indent=2),encoding='utf8')
print('PASS',len(checks),'round1 checks')
