"""Shared, serializable pipeline factory. Target transform is fitted inside each CV fold."""
import hashlib,json,platform,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from preprocess import build_pipeline
ROOT=Path(__file__).resolve().parents[1]
NAMES={'linear':'Linear Regression','tree':'Decision Tree','forest':'Random Forest','boosting':'Gradient Boosting','xgboost':'XGBoost'}
ORIGINAL={'linear':{},'tree':{'max_depth':8,'random_state':42},'forest':{'n_estimators':200,'max_depth':None,'random_state':42},'boosting':{'n_estimators':200,'learning_rate':.05,'random_state':42},'xgboost':{'n_estimators':300,'learning_rate':.05,'max_depth':4,'random_state':42,'verbosity':0}}
CLASSES={'linear':LinearRegression,'tree':DecisionTreeRegressor,'forest':RandomForestRegressor,'boosting':GradientBoostingRegressor,'xgboost':XGBRegressor}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def versions():return {'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['scikit-learn','xgboost','pandas','numpy','scipy','joblib','Flask']}}
def pipeline(slug,X,params=None,log_target=False,memory=None):
 p={**ORIGINAL[slug],**(params or {})}
 if slug in ('forest','xgboost'):p['n_jobs']=1
 estimator=CLASSES[slug](**p)
 if log_target:estimator=TransformedTargetRegressor(regressor=estimator,func=np.log1p,inverse_func=np.expm1,check_inverse=True)
 return Pipeline([('preprocess',build_pipeline(X)),('model',estimator)],memory=memory)
def unwrap(pipe):
 model=pipe.named_steps['model'];return model.regressor_ if isinstance(model,TransformedTargetRegressor) else model

def split():
 from sklearn.model_selection import train_test_split
 df=pd.read_csv(ROOT/'data/train.csv');X=df.drop(columns=['SalePrice','Id'])
 a,b,c,d=train_test_split(X,df.SalePrice,test_size=.2,random_state=42)
 return df,a,b,c,d

def save_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf8')
