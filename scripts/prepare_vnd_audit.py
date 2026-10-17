"""Read official final artifacts and export reproducible audit. Never fits a model."""
import json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'service'),str(ROOT/'notebooks_or_src_py')]
from audit_engine import split_data,metrics,sha,fit_assessment,CATALOG
from model_utils import save_json
from model_io import load_pipeline

def main():
 manifest=json.loads((ROOT/'models/final/manifest.json').read_text());selection=json.loads((ROOT/'results/tuning_report.json').read_text())
 assert sha(ROOT/'results/tuning_selection.json')==manifest['selection_sha256']
 df,Xtr,Xte,ytr,yte=split_data();indices=Xte.sample(n=30,random_state=20261015).index.tolist();demo=df.loc[indices]
 baseline={scope:metrics(y,np.repeat(ytr.mean(),len(y))) for scope,y in [('train',ytr),('test',yte)]}
 protected=['models/best_model.pkl','models/feature_defaults.json','data/train.csv','data/test.csv','results/model_results.csv','models/final/manifest.json','results/tuning_selection.json']
 report={'source_files':{f:sha(ROOT/f) for f in protected},'demo_indices':[int(i) for i in indices],'baseline':baseline,'models':{},
 'split':{'seed':42,'train_size':len(Xtr),'test_size':len(Xte),'demo_size':len(demo),'demo_seed':20261015,'train_ids':df.loc[Xtr.index,'Id'].astype(int).tolist(),'test_ids':df.loc[Xte.index,'Id'].astype(int).tolist()},
 'selection':{'winner':manifest['winner'],'cv_only':True,'selection_sha256':manifest['selection_sha256']},
 'provenance':{'dataset':'Kaggle House Prices: nhà ở Ames, Iowa; không phải bộ căn hộ Việt Nam. 79 đặc trưng, Id chỉ là khóa tra cứu.',
 'truth':'SalePrice là nhãn trong CSV gốc đã bảo toàn, chưa xác minh giao dịch bên ngoài.',
 'selection':'Cấu hình và winner chọn bằng KFold 5 folds shuffled seed 42 chỉ trên 1.168 train; holdout không tham gia lựa chọn. 30 demo cố định từ holdout.',
 'limits':'Tập holdout này đã được nhóm xem trong giai đoạn cũ; tái sử dụng để so sánh, chưa thay thế một bộ kiểm nghiệm mới độc lập.'}}
 for slug,name,_ in CATALOG:
  info=manifest['models'][slug];assert sha(ROOT/info['file'])==info['sha256'];pipe=load_pipeline(ROOT/info['file'])
  # Reproduce frozen final evaluation, never select or tune based on these values.
  scores={scope:metrics(y,pipe.predict(X)) for scope,X,y in [('train',Xtr,ytr),('test',Xte,yte)]}
  for scope in ['train','test']:
   for k in ['mae','rmse','r2']:assert np.isclose(scores[scope][k],selection['models'][slug]['after_usd'][scope][k])
  pred=pipe.predict(demo.drop(columns=['SalePrice','Id']));scores['demo']=metrics(demo.SalePrice,pred)
  report['models'][slug]={'file':info['file'],'sha256':info['sha256'],'origin':'Pipeline cuối đã bỏ Id; cấu hình chọn bằng CV trên train. Biến đổi mục tiêu: '+info['target_transform'],
   'metrics':scores,'assessment':fit_assessment(scores['train'],scores['test'],baseline),'demo_predictions':{str(int(i)):float(p) for i,p in zip(demo.Id,pred)}}
  print(name,'final holdout MAE USD',scores['test']['mae'],flush=True)
 save_json(ROOT/'results/vnd_audit_source.json',report)
 print('AUDIT EXPORTED: no fit, 79 features; final selection hash verified')
if __name__=='__main__':main()
