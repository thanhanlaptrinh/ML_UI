"""Re-evaluate saved pipelines on the original split; never call fit or save weights."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from audit_engine import ROOT, RATE, CATALOG, metrics, convert_metrics, fit_assessment, sha


def build_training_report(engine):
    original = pd.read_csv(ROOT/'results/model_results.csv').set_index('Model')
    baseline = {scope:metrics(y, np.full(len(y), engine.y_train.mean()))
                for scope,y in [('train',engine.y_train),('test',engine.y_test)]}
    tuning=json.loads((ROOT/'results/tuning_report.json').read_text())
    manifest=json.loads((ROOT/'models/final/manifest.json').read_text())
    models=[]
    for slug,name,_ in CATALOG:
        pipe=engine.models[slug]
        predictions={scope:np.asarray(pipe.predict(X),dtype=float)
                     for scope,X in [('train',engine.X_train),('test',engine.X_test)]}
        scores={scope:metrics(y,predictions[scope]) for scope,y in [('train',engine.y_train),('test',engine.y_test)]}
        y=engine.y_test.to_numpy(dtype=float); p=predictions['test']; ape=np.abs((p-y)/y)*100
        rows=[{'id':int(i),'actual':float(a)*RATE,'prediction':float(b)*RATE,
               'error':float(b-a)*RATE,'absolute_error':float(abs(b-a))*RATE,
               'percentage_error':float(e)} for i,a,b,e in zip(engine.test_ids,y,p,ape)]
        saved=original.loc[name]
        agreement=bool(abs(scores['test']['mae']-saved.MAE)<=.011 and
                       abs(scores['test']['rmse']-saved.RMSE)<=.011 and
                       abs(scores['test']['r2']-saved.R2)<=.000051)
        target=pipe.named_steps['model'];estimator=getattr(target,'regressor_',target)
        params={k:v for k,v in estimator.get_params().items()
                if k in ('n_estimators','max_depth','learning_rate','random_state')}
        models.append({'id':slug,'name':name,'metrics':{k:convert_metrics(v) for k,v in scores.items()},
            'baseline_mse_reduction_percent':float((1-scores['test']['mse']/baseline['test']['mse'])*100),
            'assessment':fit_assessment(scores['train'],scores['test'],baseline),
            'coverage':{str(t):int(np.count_nonzero(ape<=t)) for t in [5,10,20]},
            'exact_count':int(np.count_nonzero(p==y)), 'rows':rows,
            'worst':sorted(rows,key=lambda r:r['percentage_error'],reverse=True)[:5],
            'origin':engine.report['models'][slug]['origin'],
            'file':engine.report['models'][slug]['file'],'sha256':sha(ROOT/engine.report['models'][slug]['file']),
            'parameters':params,'encoded_features':int(len(pipe.named_steps['preprocess'].get_feature_names_out())),
            'original_report':{'mae':float(saved.MAE)*RATE,'rmse':float(saved.RMSE)*RATE,
                'r2':float(saved.R2),'cv_mae':float(saved.CV_MAE_mean)*RATE,
                'training_seconds':float(saved.Train_time_s), 'cv_std':float(saved.CV_MAE_std)},'matches_original_report':agreement,'tuning':tuning['models'][slug], 'target_transform':manifest['models'][slug]['target_transform']})
    winner=next(m for m in models if m['id']==manifest['winner'])
    cv_winner=min(models,key=lambda m:(m['original_report']['cv_mae'],m['original_report']['cv_std'],m['id']))
    source=ROOT/'results/repository_evidence.json'
    return {'generated_at':datetime.now(timezone.utc).isoformat(),'rate':RATE,'currency':'VND',
        'operation':'predict_only_saved_weights','models':models,'winner':winner['id'],
        'cv_winner':cv_winner['id'],'selection_rule':'MAE CV 5-fold shuffled trên train thấp nhất, hòa dùng CV std nhỏ hơn. Không dùng holdout chọn winner.',
        'split':{'train':len(engine.X_train),'holdout':len(engine.X_test),'random_state':42,'test_size':.2},
        'holdout_ids':engine.test_ids,
        'baseline':{k:convert_metrics(v) for k,v in baseline.items()},
        'source_integrity':{p:sha(ROOT/p) for p in engine.report['source_files']},
        'repository':json.loads(source.read_text()) if source.exists() else None,
        'all_match_original_report':all(m['matches_original_report'] for m in models),
        'limitations':[
            'Cấu hình cuối và winner chọn bằng CV chỉ trên train. Holdout đã được nhóm xem từ phiên bản cũ nên chưa phải bộ dữ liệu mới hoàn toàn độc lập.',
            'CV và thời gian thuộc lần tuning/train_all đã lưu; API này chỉ predict để tái lập, không chạy CV lại. Có điểm từng fold và CV std trong tuning_report.json.',
            'Trọng số gốc nằm trong models/_backup_original; cả 5 pipeline cuối được train_all.py lưu chính thức vào models/final.',
            'Dữ liệu Ames và diện tích ft²; quy đổi VND không chứng minh độ chính xác cho thị trường Việt Nam.',
            'Giá chuẩn ở đây là SalePrice trong CSV; chưa xác minh với giao dịch ngoài dữ liệu. Không bảo đảm giá căn mới đúng hoặc trong một mức sai số cố định.',
            'Chưa có khoảng dự đoán đã hiệu chuẩn, kiểm thử dữ liệu địa phương mới hoặc theo dõi dịch chuyển dữ liệu.',
            'Pipeline cuối dùng 79 đặc trưng, không có Id. Nhập thiếu đặc trưng dùng defaults từ phần train có thể làm giảm độ sát giá.'
        ]}
