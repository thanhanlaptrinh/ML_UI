import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'service'))
from audit_engine import AuditEngine, MODEL_IDS, RATE
engine=AuditEngine()
report={'overview':engine.overview(),'calculations':{s:engine.calculations(s) for s in MODEL_IDS},'source_report':engine.report}
(ROOT/'results/proof_vnd.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with (ROOT/'results/doi_chieu_30_can_VND.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['model','source_Id','actual_VND','predicted_VND','residual_VND','absolute_error_VND','squared_error_VND2','APE_percent','USD_TO_VND'])
    for slug in MODEL_IDS:
        for r in engine.calculations(slug)['rows']:
            w.writerow([slug,r['id'],r['actual'],r['prediction'],r['error'],r['absolute_error'],r['squared_error'],r['percentage_error'],RATE])
with (ROOT/'results/model_results_VND.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['Model','Scope','MAE_VND','RMSE_VND','MSE_VND2','R2','MAPE_percent','N','USD_TO_VND'])
    for model in engine.overview()['models']:
        for scope,m in model['metrics'].items():w.writerow([model['name'],scope,m['mae'],m['rmse'],m['mse'],m['r2'],m['mape'],m['n'],RATE])
print('Đã xuất proof_vnd.json, doi_chieu_30_can_VND.csv và model_results_VND.csv')

# Read-only plots and importance from final saved pipelines, with display-only VND scaling.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pipe=engine.models['xgboost'];target=pipe.named_steps['model'];estimator=getattr(target,'regressor_',target)
names=pipe.named_steps['preprocess'].get_feature_names_out()
importance=pd.DataFrame({'Feature':[str(n).removeprefix('num__').removeprefix('cat__') for n in names],'Importance':estimator.feature_importances_}).sort_values('Importance',ascending=False)
importance.to_csv(ROOT/'results/xgboost_feature_importance.csv',index=False,encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(10,7));top=importance.head(20).iloc[::-1];ax.barh(top.Feature,top.Importance);ax.set(title='XGBoost final: top 20 feature importance (no Id)',xlabel='Global feature importance');fig.tight_layout();fig.savefig(ROOT/'results/xgboost_feature_importance.png',dpi=150);plt.close(fig)
winner=engine.report['selection']['winner'];actual=engine.y_test.to_numpy()*RATE/1e9;pred=engine.models[winner].predict(engine.X_test)*RATE/1e9
fig,ax=plt.subplots(figsize=(7,6));ax.scatter(actual,pred,alpha=.6,s=20);low=min(actual.min(),pred.min());high=max(actual.max(),pred.max());ax.plot([low,high],[low,high],'--',color='black',label='Ideal prediction');ax.set(xlabel='Original SalePrice (billion VND, fixed conversion)',ylabel='Predicted price (billion VND)',title=f'Frozen CV winner: {winner} — full holdout');ax.legend();fig.tight_layout();fig.savefig(ROOT/'results/actual_vs_predicted.png',dpi=150);plt.close(fig)
from training_report import build_training_report
(ROOT/'results/training_holdout_report.json').write_text(json.dumps(build_training_report(engine),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('Exported final feature importance, holdout plot and training report; no model fitted.')
