"""Use official XGBoost model IO rather than platform-specific Booster pickle snapshots."""
import copy,hashlib,json,math
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def export_portable_xgboost(source,force=False):
    source=Path(source);manifest_path=source.parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    outputs=[source.with_suffix('.ubj'),source.with_name('xgboost_preprocess.pkl')]
    if not force and any(p.exists() for p in outputs):raise FileExistsError('Portable files exist; use --force explicitly.')
    pipe=joblib.load(source)
    target=pipe.named_steps['model'];estimator=getattr(target,'regressor_',target)
    estimator.save_model(outputs[0])
    shell=None
    if isinstance(target,TransformedTargetRegressor):
        shell=copy.copy(target);del shell.regressor_
        shell.regressor=XGBRegressor(**estimator.get_params())
    # This file contains sklearn preprocessing/optional target metadata only, no Booster.
    joblib.dump({'preprocess':pipe.named_steps['preprocess'],'target_shell':shell},outputs[1],compress=3)
    info={'source_pickle_sha256':sha(source),'format':'xgboost.save_model UBJSON',
          'parameters':{k:v for k,v in estimator.get_params().items() if not (isinstance(v,float) and not math.isfinite(v))},'files':{p.name:sha(p) for p in outputs}}
    manifest['models']['xgboost']['portable']=info
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return info

def load_pipeline(file):
    file=Path(file)
    if file.name!='xgboost.pkl':return joblib.load(file)
    manifest_path=file.parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    info=manifest['models']['xgboost'].get('portable')
    if info is None:
        raise ValueError('Thiếu XGBoost UBJ portable. Dùng gói WindowsFix đầy đủ, không chỉ chép riêng file .py.')
    if sha(file)!=info['source_pickle_sha256']:raise ValueError('XGBoost source hash mismatch.')
    for name,digest in info['files'].items():
        if sha(file.parent/name)!=digest:raise ValueError('Portable XGBoost hash mismatch: '+name)
    state=joblib.load(file.with_name('xgboost_preprocess.pkl'))
    model=XGBRegressor(**info['parameters']);model.load_model(file.with_suffix('.ubj'))
    if state['target_shell'] is not None:
        target=state['target_shell'];target.regressor_=model
    else:target=model
    return Pipeline([('preprocess',state['preprocess']),('model',target)])
