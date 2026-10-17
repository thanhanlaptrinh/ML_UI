"""Read-only prediction, currency conversion and audit explanations."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CATALOG = [
    ('linear', 'Linear Regression', 'Tổng hệ số nhân đặc trưng cộng hệ số chặn.'),
    ('tree', 'Decision Tree', 'Đi qua các điều kiện chia nhánh đến giá trị trung bình tại lá.'),
    ('forest', 'Random Forest', 'Lấy trung bình dự đoán của các cây trong rừng.'),
    ('boosting', 'Gradient Boosting', 'Giá trị khởi tạo cộng dồn các hiệu chỉnh của cây.'),
    ('xgboost', 'XGBoost', 'Cộng dồn các vòng boosting của mô hình đã lưu.')]
MODEL_IDS = [x[0] for x in CATALOG]
CONFIG = json.loads((ROOT / 'config/display.json').read_text(encoding='utf-8'))
RATE = float(os.environ.get('USD_TO_VND', CONFIG['usd_to_vnd']))
if not np.isfinite(RATE) or RATE <= 0:
    raise ValueError('USD_TO_VND phải là số dương hữu hạn.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {'mae': float(mean_absolute_error(y, p)),
            'mse': float(mean_squared_error(y, p)),
            'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'r2': float(r2_score(y, p)),
            'mape': float(np.mean(np.abs((y-p)/y))*100), 'n': len(y)}


def convert_metrics(m):
    return {k: float(v)*RATE if k in ('mae', 'rmse') else float(v)*RATE**2 if k == 'mse' else v for k, v in m.items()}


def split_data():
    df = pd.read_csv(ROOT / 'data/train.csv')
    X = df.drop(columns=['SalePrice','Id'])
    y = df.SalePrice
    a, b, c, d = train_test_split(X, y, test_size=.2, random_state=42)
    return df, a, b, c, d


def fit_assessment(train, test, baseline):
    ratio = test['mae'] / max(train['mae'], 1e-12)
    gap = train['r2'] - test['r2']
    if ratio > 1.35 and gap > .08:
        label = 'Có dấu hiệu overfit'
        detail = 'Sai số kiểm tra lớn hơn 1,35 lần sai số train và khoảng cách R² lớn hơn 0,08.'
    elif train['r2'] < .65 and test['r2'] < .65 and train['mae'] >= .75*baseline['train']['mae']:
        label = 'Có dấu hiệu underfit'
        detail = 'R² train/test đều dưới 0,65 và MAE train còn ít nhất 75% baseline dự đoán trung bình.'
    else:
        label = 'Chưa thấy dấu hiệu rõ theo ngưỡng đã đặt'
        detail = 'Không đồng thời thỏa điều kiện overfit hoặc underfit ở trên. Điều này không chứng minh mô hình tổng quát hóa hoàn hảo.'
    return {'label': label, 'detail': detail, 'test_train_mae_ratio': ratio, 'r2_gap': gap,
            'caution': 'Đây là tiêu chí chẩn đoán minh họa trên train/test tái lập; 30 mẫu riêng lẻ không đủ kết luận chắc chắn. Tập test này từng được code cũ dùng để chọn best_model nên kết quả không phải đánh giá cuối trên một tập hoàn toàn chưa từng được dùng để lựa chọn.'}


class AuditEngine:
    def __init__(self):
        self.report = json.loads((ROOT / 'results/vnd_audit_source.json').read_text(encoding='utf-8'))
        self.df, self.X_train, self.X_test, self.y_train, self.y_test = split_data()
        self.train_ids = self.df.loc[self.X_train.index,'Id'].astype(int).tolist()
        self.test_ids = self.df.loc[self.X_test.index,'Id'].astype(int).tolist()
        self.samples = self.df.loc[self.report['demo_indices']].copy()
        self.models = {}
        for slug in MODEL_IDS:
            file = ROOT / self.report['models'][slug]['file']
            if sha(file) != self.report['models'][slug]['sha256']:
                raise ValueError(f'Trọng số {slug} đã thay đổi. Chạy lại scripts/prepare_vnd_audit.py để cập nhật minh chứng.')
            from model_io import load_pipeline
            self.models[slug] = load_pipeline(file)
        for file, digest in self.report['source_files'].items():
            if sha(ROOT / file) != digest:
                raise ValueError(f'File gốc {file} đã thay đổi; không dùng lại kết quả audit cũ.')

    def overview(self):
        report = self.report
        models = []
        for slug, name, description in CATALOG:
            r = report['models'][slug]
            models.append({'id':slug, 'name':name, 'description':description,
                           'metrics': {scope:convert_metrics(value) for scope,value in r['metrics'].items()},
                           'assessment':r['assessment'], 'sha256':r['sha256'], 'origin':r['origin']})
        samples = []
        for idx, row in self.samples.iterrows():
            p = {slug:report['models'][slug]['demo_predictions'][str(int(row.Id))]*RATE for slug in MODEL_IDS}
            samples.append({'id':int(row.Id), 'actual':float(row.SalePrice)*RATE, 'predictions':p,
                            'features':{key:None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v for key,v in row.drop(['SalePrice','Id']).items()}})
        return {'models':models, 'samples':samples, 'currency':'VND', 'rate':RATE,
                'rate_note':CONFIG['rate_note'], 'provenance':report['provenance'],
                'baseline':{k:convert_metrics(v) for k,v in report['baseline'].items()},
                'integrity':report['source_files'], 'split':report['split']}

    def predict(self, slug, sample_id, changes=None):
        if slug not in self.models:
            raise ValueError('Mô hình không hợp lệ.')
        rows = self.samples[self.samples.Id == int(sample_id)]
        if rows.empty:
            raise ValueError('ID không thuộc bộ 30 mẫu cố định.')
        original = rows.iloc[0].drop(['SalePrice','Id'])
        row = original.copy()
        for key,value in (changes or {}).items():
            if key not in ('GrLivArea','LotArea','OverallQual','YearBuilt','BedroomAbvGr','FullBath'):
                raise ValueError(f'Không cho phép chỉnh {key}.')
            number = float(value)
            limits = {'GrLivArea':(100,10000),'LotArea':(100,300000),'OverallQual':(1,10),'YearBuilt':(1800,2030),'BedroomAbvGr':(0,12),'FullBath':(0,10)}
            if not np.isfinite(number) or not limits[key][0] <= number <= limits[key][1] or not number.is_integer():
                raise ValueError(f'{key}: cần số nguyên trong khoảng {limits[key]}.')
            row[key] = number
        changed = any(row[key] != original[key] for key in (changes or {}))
        frame = pd.DataFrame([row])
        pipe = self.models[slug]
        raw = float(pipe.predict(frame)[0])
        trace = self.trace(slug, pipe, frame, raw)
        actual = None if changed else float(rows.iloc[0].SalePrice)*RATE
        return {'model':slug, 'sample_id':int(sample_id), 'prediction':raw*RATE, 'currency':'VND',
                'actual':actual, 'error':None if actual is None else raw*RATE-actual,
                'changed':changed, 'trace':trace,
                'note':'Đã chỉnh đặc trưng: không có giá bán thật cho căn giả định này.' if changed else 'Nhãn giá lấy nguyên từ SalePrice của bản ghi gốc, quy đổi cùng tỷ giá với dự đoán.'}

    def trace(self, slug, pipe, frame, raw):
        # Id is accepted only as lookup metadata and is always stripped before inference.
        frame = frame.drop(columns='Id', errors='ignore')
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.pipeline import Pipeline
        target = pipe.named_steps['model']
        if isinstance(target, TransformedTargetRegressor):
            estimator = target.regressor_
            latent_pipe = Pipeline([('preprocess', pipe.named_steps['preprocess']), ('model', estimator)])
            latent_raw = float(latent_pipe.predict(frame)[0])
            latent = self.trace(slug, latent_pipe, frame, latent_raw)
            base = float(np.expm1(latent['base']/RATE))*RATE
            running = latent['base']/RATE
            converted=[]
            previous=base
            for j,step in enumerate(latent['steps']):
                if 'cumulative' in step:running=step['cumulative']/RATE
                elif slug=='forest':running=(running*j+step['amount']/RATE)/(j+1)
                elif slug=='linear':running+=step['amount']/RATE
                elif step['amount'] is not None:running=step['amount']/RATE
                else:
                    converted.append({**step,'amount':None});continue
                value=float(np.expm1(running))*RATE
                converted.append({'stage':step['stage']+' · nghịch đảo expm1(log giá)',
                                  'amount':value-previous,'cumulative':value,'log_prediction':running})
                previous=value
            reconstructed=float(np.expm1(latent_raw))*RATE
            return {**latent,'base':base,'steps':converted,'reconstructed':reconstructed,
                    'direct_prediction':raw*RATE,'difference':abs(reconstructed-raw*RATE),
                    'verified':bool(np.isclose(reconstructed,raw*RATE,atol=1,rtol=1e-6)),
                    'target_transform':'log1p → expm1',
                    'formula':'Mô hình tính z = log(1 + giá USD). Giá = expm1(z) × tỷ giá. Các giá trị lũy kế dưới đây đã nghịch đảo log; không lấy trung bình trực tiếp giá VND của các cây log.'}
        transform = pipe.named_steps['preprocess']
        T = transform.transform(frame)
        estimator = pipe.named_steps['model']
        labels = transform.get_feature_names_out()
        rows = []
        if slug == 'linear':
            dense = T.toarray()[0] if hasattr(T,'toarray') else T[0]
            contributions = np.asarray(estimator.coef_).ravel()*dense
            base = float(estimator.intercept_)
            for key,value in zip(labels,contributions):
                rows.append({'stage':str(key),'amount':float(value)*RATE})
            reconstructed = base+float(contributions.sum())
            formula = 'ŷ = intercept + Σ(coefⱼ × xⱼ sau tiền xử lý). Bảng liệt kê mọi đóng góp, kể cả đóng góp âm.'
        elif slug == 'tree':
            dense = T.toarray() if hasattr(T,'toarray') else T
            tree = estimator.tree_
            path = estimator.decision_path(T).indices
            leaf = int(estimator.apply(T)[0])
            for node in path:
                f = tree.feature[node]
                if f >= 0:
                    value = float(dense[0,f]);threshold = float(tree.threshold[node])
                    rows.append({'stage':f'Nút {node}: {labels[f]} = {value:.5g} {"≤" if value<=threshold else ">"} {threshold:.5g}', 'amount':None})
            base = float(tree.value[leaf].ravel()[0]);reconstructed=base
            rows.append({'stage':f'Lá {leaf}, giá trị dự đoán trung bình lá','amount':base*RATE})
            formula = 'Theo các nhánh từ gốc đến lá; giá trị tại lá là dự đoán của cây.'
        elif slug == 'forest':
            values = [float(tree.predict(T)[0]) for tree in estimator.estimators_]
            base=0.;reconstructed=float(np.mean(values))
            rows=[{'stage':f'Cây {i+1}', 'amount':v*RATE} for i,v in enumerate(values)]
            formula=f'ŷ = (giá cây 1 + … + giá cây {len(values)}) / {len(values)}. Các cây có trọng số bằng nhau.'
        else:
            if slug == 'boosting':
                base=float(estimator.init_.predict(T)[0])
                stages=[float(v[0]) for v in estimator.staged_predict(T)]
            else:
                config=json.loads(estimator.get_booster().save_config())
                base_value=config['learner']['learner_model_param']['base_score']
                base=float(json.loads(base_value)[0] if base_value.startswith('[') else base_value)
                stages=[float(estimator.predict(T,iteration_range=(0,k))[0]) for k in range(1,estimator.get_booster().num_boosted_rounds()+1)]
            previous=base
            for i,value in enumerate(stages):
                rows.append({'stage':f'Vòng {i+1}','amount':(value-previous)*RATE,'cumulative':value*RATE})
                previous=value
            reconstructed=previous
            formula='ŷ = giá khởi tạo + Σ hiệu chỉnh từng vòng. Hiệu chỉnh đã bao gồm learning_rate; tổng lũy kế lấy trực tiếp từ các vòng của mô hình.'
        return {'input_features':int(frame.shape[1]), 'missing_values':int(frame.isna().sum().sum()),
                'encoded_features':int(T.shape[1]), 'formula':formula, 'base':base*RATE,
                'reconstructed':reconstructed*RATE, 'direct_prediction':raw*RATE,
                'difference':abs(reconstructed-raw)*RATE, 'steps':rows,
                'verified':bool(np.isclose(reconstructed,raw,atol=.02,rtol=1e-6))}

    def calculations(self, slug):
        if slug not in MODEL_IDS:
            raise ValueError('Mô hình không hợp lệ.')
        y=self.samples.SalePrice.to_numpy(dtype=float)*RATE
        p=np.array([self.report['models'][slug]['demo_predictions'][str(int(i))] for i in self.samples.Id])*RATE
        e=p-y;mean=float(y.mean());sst=float(np.square(y-mean).sum())
        rows=[{'id':int(i),'actual':float(a),'prediction':float(b),'error':float(c),'absolute_error':float(abs(c)),
               'squared_error':float(c*c),'percentage_error':float(abs(c)/a*100)} for i,a,b,c in zip(self.samples.Id,y,p,e)]
        return {'rows':rows,'n':len(y),'sum_absolute_error':float(abs(e).sum()),'sse':float(np.square(e).sum()),
                'mean_actual':mean,'sst':sst,'sum_percentage_error':float(np.sum(abs(e)/y*100)),
                'metrics':metrics(y,p)}
