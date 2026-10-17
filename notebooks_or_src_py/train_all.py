"""Fit all frozen CV selections, then score holdout. Explicit --force required to replace artifacts."""
import argparse,json,sys,time,shutil,os
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from model_utils import ROOT,NAMES,pipeline,split,versions,digest,save_json
sys.path.insert(0,str(ROOT/'scripts'))
from backup_original import backup
from export_defaults import compute_defaults

def score(y,p):
 y,p=np.asarray(y,float),np.asarray(p,float)
 return {'mae':float(mean_absolute_error(y,p)),'mse':float(mean_squared_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'r2':float(r2_score(y,p)),'mape':float(np.mean(abs(y-p)/y)*100),'n':len(y)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');args=ap.parse_args()
 selection_path=ROOT/'results/tuning_selection.json';selection=json.loads(selection_path.read_text());assert set(selection['models'])==set(NAMES) and selection['winner'] in NAMES
 selected_report=json.loads((ROOT/'results/tuning_report.json').read_text())
 assert selected_report['selection_sha256']==digest(selection_path)
 assert all(selected_report['models'][s]['selected']==selection['models'][s]['selected'] for s in NAMES), 'Report selections differ from frozen train-only selection'
 expected=min(NAMES,key=lambda s:(selection['models'][s]['selected']['cv_mae'],selection['models'][s]['selected']['cv_std'],s));assert expected==selection['winner']
 outputs=[ROOT/'models/final'/f'{s}.pkl' for s in NAMES]+[ROOT/'models/best_model.pkl',ROOT/'models/feature_defaults.json',ROOT/'results/model_results.csv']
 if not args.force and any(p.exists() for p in outputs):raise FileExistsError('Outputs already exist. Back up first, then explicitly use --force. Nothing was trained or overwritten.')
 baseline=backup()
 assert all(digest(ROOT/p)==h for p,h in baseline['data_hashes'].items())
 df,Xtr,Xte,ytr,yte=split();assert Xtr.shape[1]==79 and list(Xtr.columns)==selection['features']
 assert selection['train_indices']==[int(i) for i in Xtr.index]
 staging=ROOT/'.cache/final_staging';staging.mkdir(parents=True,exist_ok=True)
 manifest={'created_at':datetime.now(timezone.utc).isoformat(),'seed':42,'winner':expected,'selection_sha256':digest(selection_path),'selection_rule':selection['selection_rule'],'features':list(Xtr.columns),'input_features':Xtr.shape[1],'data_hashes':baseline['data_hashes'],'versions':versions(),'models':{},'holdout_used_for_selection':False,'holdout_policy':'All selections frozen before the single final batch of holdout predictions. Subsequent verify/API inference only reproduces the frozen results; it never triggers tuning.'}
 fitted={};durations={}
 for slug in NAMES:
  chosen=selected_report['models'][slug]['selected'];start=time.perf_counter();pipe=pipeline(slug,Xtr,chosen['parameters'],chosen['variant']=='log1p');pipe.fit(Xtr,ytr);durations[slug]=time.perf_counter()-start
  assert 'Id' not in pipe.feature_names_in_ and len(pipe.feature_names_in_)==79
  joblib.dump(pipe,staging/f'{slug}.pkl',compress=3);fitted[slug]=pipe
  manifest['models'][slug]={'file':f'models/final/{slug}.pkl','sha256':digest(staging/f'{slug}.pkl'),'bytes':(staging/f'{slug}.pkl').stat().st_size,'target_transform':chosen['variant'],'parameters':chosen['parameters'],'cv_mae':chosen['cv_mae'],'cv_std':chosen['cv_std'],'train_seconds':durations[slug]}
 # Selection is already frozen. This is the only evaluation phase of this training run.
 report={**selection,'selection_sha256':manifest['selection_sha256'],'final_manifest':'models/final/manifest.json','holdout_used_for_selection':False,'final_evaluation_at':datetime.now(timezone.utc).isoformat(),'stage':'final_evaluation_complete'}
 oldaudit=json.loads((ROOT/'results/baseline_original/vnd_audit_source.json').read_text());oldtable=pd.read_csv(ROOT/'results/baseline_original/model_results.csv').set_index('Model')
 rate=json.loads((ROOT/'config/display.json').read_text())['usd_to_vnd'];summaries=[];current=[]
 for slug,name in NAMES.items():
  train=score(ytr,fitted[slug].predict(Xtr));test=score(yte,fitted[slug].predict(Xte));old=oldaudit['models'][slug]['metrics'];row=report['models'][slug]
  row.update({'best_params':{**__import__('model_utils').ORIGINAL[slug],**row['selected']['parameters']},'before_usd':{k:old[k] for k in ['train','test']},'after_usd':{'train':train,'test':test},'after_vnd':{scope:{k:v*rate if k in ('mae','rmse') else v*rate**2 if k=='mse' else v for k,v in metric.items()} for scope,metric in [('train',train),('test',test)]},'before_vnd':{scope:{k:v*rate if k in ('mae','rmse') else v*rate**2 if k=='mse' else v for k,v in old[scope].items()} for scope in ['train','test']},'holdout_train_mae_ratio':test['mae']/train['mae'],'r2_gap':train['r2']-test['r2'],'original_csv_cv_mae':float(oldtable.loc[name,'CV_MAE_mean']),'final_fit_seconds':durations[slug]})
  summaries.append({'Model':name,'Variant':row['selected']['variant'],'CV_before_with_Id_USD':row['original_with_Id_cv']['mae'],'CV_control_no_Id_USD':row['original_without_Id_cv']['mae'],'CV_after_USD':row['selected']['cv_mae'],'CV_std_USD':row['selected']['cv_std'],'Holdout_MAE_before_USD':old['test']['mae'],'Holdout_MAE_after_USD':test['mae'],'Holdout_RMSE_before_USD':old['test']['rmse'],'Holdout_RMSE_after_USD':test['rmse'],'Holdout_R2_before':old['test']['r2'],'Holdout_R2_after':test['r2'],'Holdout_MAE_before_VND':old['test']['mae']*rate,'Holdout_MAE_after_VND':test['mae']*rate,'Train_MAE_USD':train['mae'],'Train_R2':train['r2'],'Holdout_Train_MAE_ratio':test['mae']/train['mae'],'R2_gap':train['r2']-test['r2'],'Winner_by_CV':slug==expected})
  current.append({'Model':name,'MAE':test['mae'],'RMSE':test['rmse'],'R2':test['r2'],'CV_MAE_mean':row['selected']['cv_mae'],'CV_MAE_std':row['selected']['cv_std'],'Train_time_s':durations[slug],'Variant':row['selected']['variant']})
 directory=ROOT/'models/final';directory.mkdir(parents=True,exist_ok=True)
 for slug in NAMES:os.replace(staging/f'{slug}.pkl',directory/f'{slug}.pkl')
 shutil.copy2(directory/f'{expected}.pkl',ROOT/'models/best_model.pkl')
 manifest['best_model']={'file':'models/best_model.pkl','model':expected,'sha256':digest(ROOT/'models/best_model.pkl')}
 save_json(directory/'manifest.json',manifest)
 sys.path.insert(0,str(ROOT/'service'))
 from model_io import export_portable_xgboost
 export_portable_xgboost(directory/'xgboost.pkl',force=args.force)
 save_json(ROOT/'models/best_model.manifest.json',manifest['best_model'])
 save_json(ROOT/'models/feature_defaults.json',compute_defaults(Xtr))
 save_json(ROOT/'results/tuning_report.json',report)
 pd.DataFrame(summaries).to_csv(ROOT/'results/tuning_summary.csv',index=False,encoding='utf-8-sig');pd.DataFrame(current).to_csv(ROOT/'results/model_results.csv',index=False,encoding='utf-8-sig')
 assert all(digest(ROOT/f['backup'])==f['sha256'] for f in baseline['files'])
 print(pd.DataFrame(summaries).to_string(index=False));print('CV WINNER:',expected,'| ORIGINAL BACKUPS VERIFIED | train/test CSV UNCHANGED')
if __name__=='__main__':main()
