"""Train-only 5-fold search; NO holdout prediction/score or selection in this script."""
import argparse,json,sys,time,os
from pathlib import Path
import joblib,numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold,GridSearchCV,RandomizedSearchCV,cross_val_score,train_test_split
from model_utils import ROOT,NAMES,ORIGINAL,pipeline,split,save_json,versions,digest
sys.path.insert(0,str(ROOT/'scripts'))
from backup_original import backup
# Deliberately smaller bounds than the suggested exhaustive spaces; exactly 40 random candidates per variant.
SPACES={
 'tree':{'max_depth':[3,4,5,6,8,10,12,None],'min_samples_leaf':[1,2,4,8,16],'min_samples_split':[2,5,10,20]},
 'forest':{'n_estimators':[200,300],'max_depth':[None,8,12,16],'min_samples_leaf':[1,2,3,5],'min_samples_split':[2,5,10],'max_features':['sqrt',.5,.7,1.]},
 'boosting':{'n_estimators':[200,400],'learning_rate':[.03,.05,.08],'max_depth':[2,3,4],'subsample':[.7,.9,1.],'min_samples_leaf':[1,3,5,10],'max_features':[None,.5,.8]},
 'xgboost':{'n_estimators':[300,500,800],'learning_rate':[.02,.03,.05],'max_depth':[2,3,4],'min_child_weight':[3,5,8],'subsample':[.7,.9,1.],'colsample_bytree':[.5,.7,1.],'reg_lambda':[1,5,10,20],'reg_alpha':[0,.5,1],'gamma':[0,.5,1]}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 out=ROOT/'results/tuning_selection.json'
 if out.exists() and not args.force and not args.resume:raise FileExistsError('Selection exists; use --resume or --force explicitly')
 original=backup();df,X,_,y,_=split();assert X.shape==(1168,79) and 'Id' not in X
 cache=ROOT/'.cache/tuning';cache.mkdir(parents=True,exist_ok=True)
 cv=KFold(n_splits=5,shuffle=True,random_state=42);memory=joblib.Memory(cache/'preprocess',verbose=0)
 history={'seed':42,'cv':{'folds':5,'shuffle':True,'scoring':'neg_mean_absolute_error','n_jobs':-1},'train_indices':[int(i) for i in X.index],
 'train_ids':[int(i) for i in df.loc[X.index,'Id']],'features':list(X.columns),'versions':versions(),'search_spaces':SPACES,
 'search_limits_reason':'Bounded to RF 200/300 trees, GB 200/400 rounds and XGB 300/500/800 rounds, depths mostly 2–4, to limit CPU cost. 40 random configurations per raw/log variant retained. No holdout-based changes.',
 'selection_rule':'Minimum train-only 5-fold CV MAE in USD; tie broken by smaller fold standard deviation, then model ID. Raw/log chosen by same CV only.',
 'models':{},'data_hashes':original['data_hashes']}
 original_paths={'xgboost':'models/_backup_original/best_model.pkl',**{s:f'models/_backup_original/vnd_supplement/{s}.pkl' for s in ['linear','tree','forest','boosting']}}
 with joblib.parallel_config(backend='loky',inner_max_num_threads=1):
  for slug,name in NAMES.items():
   checkpoint=cache/f'{slug}_selection.json'
   if args.resume and checkpoint.exists():history['models'][slug]=json.loads(checkpoint.read_text());print('RESUME',slug,flush=True);continue
   start=time.perf_counter();print('START',name,flush=True)
   old=joblib.load(ROOT/original_paths[slug]);Xold=df.loc[X.index].drop(columns='SalePrice')
   old_scores=-cross_val_score(clone(old),Xold,y,cv=cv,scoring='neg_mean_absolute_error',n_jobs=-1)
   base=pipeline(slug,X,memory=memory)
   control=-cross_val_score(base,X,y,cv=cv,scoring='neg_mean_absolute_error',n_jobs=-1)
   candidates=[{'variant':'raw','parameters':ORIGINAL[slug],'cv_mae':float(control.mean()),'cv_std':float(control.std()),'kind':'original_parameters_without_Id','fold_mae':control.tolist()}];searches=[]
   if slug!='linear':
    for log in ([False,True] if slug in ('forest','boosting','xgboost') else [False]):
     prefix='model__regressor__' if log else 'model__'
     space={prefix+k:v for k,v in SPACES[slug].items()}
     factory=GridSearchCV if slug=='tree' else RandomizedSearchCV
     extra={} if slug=='tree' else {'n_iter':40,'random_state':42}
     search=factory(pipeline(slug,X,log_target=log,memory=memory),space,cv=cv,scoring='neg_mean_absolute_error',refit=True,n_jobs=-1,error_score='raise',return_train_score=True,pre_dispatch='2*n_jobs',**extra)
     print('SEARCH',slug,'log1p' if log else 'raw',flush=True);search.fit(X,y)
     results=[]
     for j,p in enumerate(search.cv_results_['params']):
      results.append({'variant':'log1p' if log else 'raw','parameters':{k.removeprefix(prefix):v for k,v in p.items()},'cv_mae':float(-search.cv_results_['mean_test_score'][j]),'cv_std':float(search.cv_results_['std_test_score'][j]),'cv_train_mae':float(-search.cv_results_['mean_train_score'][j]),'fold_mae':[float(-search.cv_results_[f'split{k}_test_score'][j]) for k in range(5)],'kind':'grid' if slug=='tree' else 'randomized'})
     ordered=sorted(results,key=lambda c:(c['cv_mae'],c['cv_std']))
     searches.append({'variant':'log1p' if log else 'raw','n_iter':len(results),'top5':ordered[:5],'best_score_':float(search.best_score_),'refit':True})
     candidates+=results
     save_json(cache/f'{slug}_{"log" if log else "raw"}_cv_results.json',results)
     archive=ROOT/'results/tuning_cv_results';archive.mkdir(parents=True,exist_ok=True)
     save_json(archive/f'{slug}_{"log" if log else "raw"}.json',results)
   # Early stopping is an independent train-only candidate, never triggered by holdout.
   if slug=='xgboost':
    chosen=min(candidates,key=lambda c:(c['cv_mae'],c['cv_std']))
    trial={**ORIGINAL[slug],**chosen['parameters'],'min_child_weight':8,'reg_lambda':20,'subsample':.8,'max_depth':min(3,chosen['parameters'].get('max_depth',3)),'n_estimators':1200,'early_stopping_rounds':40,'n_jobs':1}
    from xgboost import XGBRegressor
    from preprocess import build_pipeline
    # For each outer train fold, select iteration ONLY from an inner training/validation split.
    fold_scores=[];iterations=[]
    for tr,va in cv.split(X):
     a,b=train_test_split(tr,test_size=.2,random_state=42)
     prep=build_pipeline(X.iloc[a]);A=prep.fit_transform(X.iloc[a]);B=prep.transform(X.iloc[b]);model=XGBRegressor(**trial)
     log=chosen['variant']=='log1p';transform=np.log1p if log else lambda v:v
     model.fit(A,transform(y.iloc[a]),eval_set=[(B,transform(y.iloc[b]))],verbose=False)
     iteration=int(model.best_iteration)+1;iterations.append(iteration)
     # refit outer fold using only its inner-selected iteration count.
     params={k:v for k,v in trial.items() if k!='early_stopping_rounds'};params['n_estimators']=iteration
     pipe=pipeline(slug,X,params,log);pipe.fit(X.iloc[tr],y.iloc[tr]);p=pipe.predict(X.iloc[va]);fold_scores.append(float(np.mean(abs(p-y.iloc[va]))))
    a,b=train_test_split(np.arange(len(X)),test_size=.2,random_state=42);prep=build_pipeline(X.iloc[a]);A=prep.fit_transform(X.iloc[a]);B=prep.transform(X.iloc[b]);es=XGBRegressor(**trial);es.fit(A,transform(y.iloc[a]),eval_set=[(B,transform(y.iloc[b]))],verbose=False)
    params={k:v for k,v in trial.items() if k!='early_stopping_rounds'};params['n_estimators']=int(es.best_iteration)+1
    early={'variant':chosen['variant'],'parameters':params,'cv_mae':float(np.mean(fold_scores)),'cv_std':float(np.std(fold_scores)),'kind':'nested_train_only_early_stopping','fold_mae':fold_scores,'fold_iterations':iterations,'full_train_inner_best_iteration':int(es.best_iteration)+1,'validation_fraction':.2,'early_stopping_rounds':40}
    candidates.append(early);searches.append({'variant':'regularized_inner_early_stopping','n_iter':1,'top5':[early]})
   best=min(candidates,key=lambda c:(c['cv_mae'],c['cv_std']))
   result={'id':slug,'name':name,'original_parameters':ORIGINAL[slug],'original_with_Id_cv':{'mae':float(old_scores.mean()),'std':float(old_scores.std()),'fold_mae':old_scores.tolist()},'original_without_Id_cv':{'mae':float(control.mean()),'std':float(control.std()),'fold_mae':control.tolist()},'selected':best,'top5':sorted(candidates,key=lambda c:(c['cv_mae'],c['cv_std']))[:5],'searches':searches,'runtime_seconds':time.perf_counter()-start,'n_iter':sum(s['n_iter'] for s in searches),'seed':42}
   history['models'][slug]=result;save_json(checkpoint,result);save_json(out,history)
   print('DONE',slug,'old CV',old_scores.mean(),'no-ID control',control.mean(),'selected CV',best['cv_mae'],'variant',best['variant'],flush=True)
 winner=min(history['models'],key=lambda s:(history['models'][s]['selected']['cv_mae'],history['models'][s]['selected']['cv_std'],s));history['winner']=winner
 # Selection file contains no holdout scores; hash is pinned in final manifest before holdout is read.
 save_json(out,history)
 save_json(ROOT/'results/tuning_report.json',{**history,'stage':'cv_frozen_pending_final_evaluation','selection_sha256':digest(out)})
 import pandas as pd
 pd.DataFrame([{'Model':m['name'],'CV_before_with_Id_USD':m['original_with_Id_cv']['mae'],'CV_control_no_Id_USD':m['original_without_Id_cv']['mae'],'CV_after_USD':m['selected']['cv_mae'],'CV_std_USD':m['selected']['cv_std'],'Variant':m['selected']['variant'],'Winner_by_CV':s==winner} for s,m in history['models'].items()]).to_csv(ROOT/'results/tuning_summary.csv',index=False,encoding='utf-8-sig')
 print('FROZEN CV WINNER:',winner,'selection SHA256',digest(out),flush=True)
if __name__=='__main__':main()
