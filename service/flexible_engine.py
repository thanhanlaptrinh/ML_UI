"""Dynamic inference on arbitrary rows. Never fabricates ground truth or trains weights."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from audit_engine import RATE, ROOT, CATALOG, MODEL_IDS, convert_metrics


class FlexibleEngine:
    def __init__(self, audit):
        self.audit = audit
        self.defaults = json.loads((ROOT/'models/feature_defaults.json').read_text(encoding='utf-8'))
        self.columns = list(audit.X_train.columns)
        self.numeric = list(audit.X_train.select_dtypes(include='number').columns)
        self.categorical = [c for c in self.columns if c not in self.numeric]
        self.index = audit.df.set_index('Id', drop=False)
        self.category_values={c:set(str(v) for v in self.audit.df[c].dropna().unique()) for c in self.categorical}
        self.ranges={c:(float(self.audit.df[c].min()),float(self.audit.df[c].max())) for c in self.numeric}
        self.train_ids = set(audit.train_ids)
        self.baseline_value = float(audit.y_train.mean())*RATE

    def schema(self):
        return {'defaults':self.defaults,'numeric':self.numeric,'categorical':self.categorical,
                'categories':{c:sorted(str(v) for v in self.audit.df[c].dropna().unique()) for c in self.categorical},
                'columns':self.columns,'rate':RATE,'baseline_prediction':self.baseline_value,
                'source_ids':[int(i) for i in self.audit.df.Id], 'limits':{'max_rows':2000},
                'truth_policy':'Giá gốc được kiểm tra bằng ID và các đặc trưng. Giá nhập tay/CSV mới là nhãn người dùng cung cấp, không được xác minh giao dịch bên ngoài.'}

    def dataset(self, kind):
        if kind=='demo':
            return {'records':[{'source_id':int(i)} for i in self.audit.samples.Id], 'name':'30 căn holdout cố định'}
        if kind=='kaggle-test':
            test=pd.read_csv(ROOT/'data/test.csv')
            return {'records':[{'id':str(int(row.Id)),'features':self.clean(row.drop('Id').to_dict())} for _,row in test.iterrows()], 'name':'Kaggle test.csv: không có nhãn SalePrice'}
        raise ValueError('Bộ dữ liệu không hợp lệ.')

    def lookup(self, value):
        try:
            number=float(value)
            if not number.is_integer():raise ValueError()
            row=self.index.loc[int(number)]
        except (ValueError,TypeError,KeyError):
            raise ValueError('Không tìm thấy ID trong data/train.csv có nhãn giá. ID test.csv chưa có giá thực tế.')
        return {'source_id':int(number),'features':self.clean(row.drop(['SalePrice','Id']).to_dict()),
                'actual_vnd':float(row.SalePrice)*RATE,
                'membership':'train' if int(number) in self.train_ids else 'holdout',
                'provenance':'Nhãn SalePrice khớp CSV gốc đã kiểm tra SHA-256; chưa xác minh giao dịch bên ngoài.'}

    @staticmethod
    def clean(obj):
        return {k:None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v for k,v in obj.items()}

    @staticmethod
    def same(a,b):
        if pd.isna(a) and pd.isna(b):return True
        if pd.isna(a) or pd.isna(b):return False
        if isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
            return bool(np.isclose(a,b,atol=1e-10,rtol=1e-12))
        return str(a)==str(b)

    def prepare(self, records, target_currency):
        if not isinstance(records,list) or not 1<=len(records)<=2000:
            raise ValueError('Cần từ 1 đến 2.000 bản ghi.')
        if target_currency not in ('USD','VND'):raise ValueError('Đơn vị nhãn chỉ nhận USD hoặc VND.')
        frames=[];meta=[]
        for i,record in enumerate(records):
            if not isinstance(record,dict):raise ValueError(f'Dòng {i+1}: cần đối tượng JSON.')
            fields=record.get('features',{})
            if not isinstance(fields,dict):raise ValueError(f'Dòng {i+1}: features phải là đối tượng JSON.')
            source_id=record.get('source_id')
            source=self.lookup(source_id) if source_id is not None else None
            row=dict(source['features'] if source else self.defaults)
            unknown=set(fields)-set(self.columns)-{'Id'}
            if unknown:raise ValueError(f'Dòng {i+1}: cột đặc trưng không hợp lệ: {", ".join(sorted(unknown))}.')
            if not source and not (set(fields)-{'Id'}):raise ValueError(f'Dòng {i+1}: cần đặc trưng nhà ngoài Id; sample_submission không phải dữ liệu có nhãn thật.')
            row.update({k:v for k,v in fields.items() if k!='Id'})
            warnings=[]
            for key in self.columns:
                value=row[key]
                if value is None or value=='':row[key]=np.nan;continue
                if key in self.numeric:
                    if isinstance(value,bool):raise ValueError(f'Dòng {i+1}: {key} không nhận boolean.')
                    try: number=float(value)
                    except (ValueError,TypeError):raise ValueError(f'Dòng {i+1}: {key} phải là số.')
                    if not np.isfinite(number) or abs(number)>1e12:raise ValueError(f'Dòng {i+1}: {key} phải là số hữu hạn.')
                    if key=='OverallQual' and not (1<=number<=10):raise ValueError('OverallQual cần nằm trong 1–10.')
                    if key in ('LotArea','GrLivArea') and number<=0:raise ValueError(f'{key} cần lớn hơn 0.')
                    row[key]=number
                    low,high=self.ranges[key]
                    if number<low or number>high:warnings.append(f'{key} ngoài miền dữ liệu huấn luyện.')
                else:
                    if not isinstance(value,(str,int,float)):raise ValueError(f'Dòng {i+1}: {key} cần giá trị phân loại.')
                    row[key]=str(value)
                    if row[key] not in self.category_values[key]:warnings.append(f'{key}: nhãn mới, One-Hot có thể bỏ qua.')
            actual=record.get('actual_vnd')
            if actual is None or actual=='':
                actual=record.get('actual')
                if actual is not None and actual!='':
                    try:actual=float(actual)*(RATE if target_currency=='USD' else 1)
                    except (ValueError,TypeError):raise ValueError(f'Dòng {i+1}: giá thực tế không phải số.')
            truth='unavailable';membership='unknown';matched_id=None
            # Explicit lookup may verify an unmodified full row. Imported rows must provide ALL columns.
            candidate=source_id if source else fields.get('Id') if set(self.columns)<=set(fields) else None
            if candidate is not None:
                try:
                    original=self.lookup(candidate)
                    unchanged=all(self.same(row[k],original['features'][k]) for k in self.columns)
                    if unchanged:
                        matched_id=original['source_id'];membership=original['membership']
                        if actual is None or actual=='':actual=original['actual_vnd'];truth='original_csv'
                        elif np.isclose(float(actual),original['actual_vnd'],atol=.01,rtol=1e-10):truth='original_csv'
                        else:warnings.append('Nhãn nhập khác SalePrice gốc của cùng căn. Dùng nhãn người dùng, không gọi là nhãn gốc.')
                except (ValueError,TypeError):pass
            if actual is None or actual=='':actual=None
            else:
                try:actual=float(actual)
                except (ValueError,TypeError):raise ValueError(f'Dòng {i+1}: giá thực tế cần là số.')
                if not np.isfinite(actual) or actual<=0 or actual>1e16:raise ValueError(f'Dòng {i+1}: giá thực tế phải dương, hữu hạn.')
                if truth!='original_csv':truth='user_supplied'
            defaulted=[] if source else [k for k in self.columns if k not in fields]
            if defaulted:warnings.append(f'{len(defaulted)} cột chưa nhập dùng giá trị mặc định đã lưu; đây là đầu vào xấp xỉ.')
            frames.append(row)
            meta.append({'id':str(record.get('id',source_id if source_id is not None else fields.get('Id',i+1))),
                         'actual':actual,'truth':truth,'membership':membership,'matched_source_id':matched_id,
                         'defaulted_columns':defaulted,'warnings':warnings})
        X=pd.DataFrame(frames,columns=self.columns)
        for col in self.numeric:X[col]=pd.to_numeric(X[col],errors='raise')
        return X,meta

    @staticmethod
    def score(y,p):
        y,p=np.asarray(y,float),np.asarray(p,float)
        if not len(y):return None
        mse=float(mean_squared_error(y,p));sst=float(np.square(y-y.mean()).sum())
        return {'n':len(y),'mae':float(mean_absolute_error(y,p)),'mse':mse,'rmse':float(np.sqrt(mse)),
                'r2':float(r2_score(y,p)) if len(y)>1 and sst>1e-12 else None,
                'mape':float(np.mean(abs(y-p)/y)*100),
                'sum_absolute_error':float(abs(y-p).sum()),'sse':float(np.square(y-p).sum()),
                'sst':sst,'mean_actual':float(y.mean()),'sum_percentage_error':float(np.sum(abs(y-p)/y*100))}

    def weights(self,slug,pipe,frame):
        target=pipe.named_steps['model'];model=getattr(target,'regressor_',target);prep=pipe.named_steps['preprocess'];T=prep.transform(frame)
        feature_names=prep.get_feature_names_out()
        common={'sha256':self.audit.report['models'][slug]['sha256'], 'description':'Tham chiếu từ đúng mô hình đã nạp, không fit lại.' + (' Mô hình học log1p(giá USD), nghịch đảo expm1 trước khi quy đổi VND.' if hasattr(target,'regressor_') else ''), 'items':[]}
        if slug=='linear':
            values=T.toarray()[0] if hasattr(T,'toarray') else T[0]
            coeff=np.asarray(model.coef_).ravel();contrib=coeff*values
            top=np.argsort(abs(contrib))[-12:][::-1]
            common.update({'kind':'Hệ số tuyến tính','intercept_vnd':float(model.intercept_)*RATE,
                'items':[{'feature':str(feature_names[j]),'input':float(values[j]),'weight_vnd':float(coeff[j])*RATE,'contribution_vnd':float(contrib[j])*RATE} for j in top]})
        else:
            if slug=='tree':
                leaf=int(model.apply(T)[0]);common.update({'kind':'Cấu trúc cây','leaf':leaf,'leaf_train_samples':int(model.tree_.n_node_samples[leaf]),'depth':int(model.get_depth())})
            elif slug=='forest':common.update({'kind':'Trọng số tổng hợp cây','trees':len(model.estimators_),'weight_per_tree':1/len(model.estimators_)})
            else:common.update({'kind':'Boosting','learning_rate':float(model.learning_rate),'stages':int(model.n_estimators),'max_depth':int(model.max_depth)})
            importance=np.asarray(model.feature_importances_);top=np.argsort(importance)[-12:][::-1]
            common['items']=[{'feature':str(feature_names[j]),'importance':float(importance[j])} for j in top]
            common['importance_note']='Feature importance toàn mô hình, không phải xác suất đúng hoặc đóng góp nhân quả riêng cho căn này.'
        return common

    def evaluate(self,payload):
        if not isinstance(payload,dict):raise ValueError('Cần đối tượng JSON.')
        selected=payload.get('models',MODEL_IDS)
        if not isinstance(selected,list) or not selected or len(set(selected))!=len(selected) or any(slug not in MODEL_IDS for slug in selected):
            raise ValueError('Chọn danh sách mô hình hợp lệ, không trùng lặp.')
        X,rows=self.prepare(payload.get('records'),payload.get('target_currency','VND'))
        mask=np.array([r['actual'] is not None for r in rows]);y=np.array([r['actual'] for r in rows if r['actual'] is not None],float)
        baseline=self.score(y,np.repeat(self.baseline_value,len(y)))
        result=[]
        for slug in selected:
            pipe=self.audit.models[slug];pred=np.asarray(pipe.predict(X),float)*RATE
            if not np.isfinite(pred).all():raise ValueError(f'{slug} trả về số không hữu hạn. Kiểm tra dữ liệu.')
            score=self.score(y,pred[mask]);details=[]
            for row,p in zip(rows,pred):
                error=None if row['actual'] is None else float(p-row['actual'])
                details.append({'prediction':float(p),'error':error,'absolute_error':None if error is None else abs(error),
                                'squared_error':None if error is None else error**2,
                                'percentage_error':None if error is None else abs(error)/row['actual']*100})
            r={'id':slug,'name':next(n for s,n,_ in CATALOG if s==slug),'predictions':details,'metrics':score,
               'baseline_mse_delta':None if score is None else score['mse']-baseline['mse'],
               'mse_reduction_percent':None if score is None or baseline['mse']<=0 else (1-score['mse']/baseline['mse'])*100,
               'original_fit_assessment':self.audit.report['models'][slug]['assessment'],
               'original_train_test':{scope:convert_metrics(value) for scope,value in self.audit.report['models'][slug]['metrics'].items() if scope in ('train','test')}}
            if payload.get('explain'):
                if len(rows)!=1:raise ValueError('Giải thích chi tiết chỉ nhận một bản ghi mỗi lần.')
                raw=float(pred[0]/RATE);r['trace']=self.audit.trace(slug,pipe,X,raw);r['weights']=self.weights(slug,pipe,X)
                actual=rows[0]['actual'];running=r['trace']['base'];stages=[]
                for j,step in enumerate(r['trace']['steps']):
                    if 'cumulative' in step:running=step['cumulative']
                    elif slug=='forest':
                        running=(running*j+step['amount'])/(j+1)
                    elif slug=='linear':running+=step['amount']
                    elif 'cumulative' in step:running=step['cumulative']
                    elif step['amount'] is not None:running=step['amount']
                    else:continue
                    stages.append({'stage':j+1,'prediction':float(running),'squared_error':None if actual is None else float((running-actual)**2)})
                r['stage_evaluation']=stages
                r['stage_note']='Giá trị của các phần mô hình khi suy luận, không phải lịch sử training loss. Thứ tự cộng đóng góp tuyến tính không phải quá trình tối ưu trọng số.'
            result.append(r)
        label_count=int(mask.sum());best=None
        if label_count:
            best=[r['id'] for r in result if np.isclose(r['metrics']['mae'],min(m['metrics']['mae'] for m in result),atol=1e-5,rtol=1e-12)]
        encoded=X.to_json(orient='split',double_precision=15)+json.dumps([r['actual'] for r in rows])
        return {'rows':rows,'models':result,'count':len(rows),'labeled_count':label_count,'unlabeled_count':len(rows)-label_count,
                'best_by_mae':best,'baseline':baseline,'baseline_prediction':self.baseline_value,
                'currency':'VND','rate':RATE,'evaluation_hash':hashlib.sha256(encoded.encode()).hexdigest(),
                'fit_note':'Không kết luận overfit/underfit từ một căn hoặc một bộ test mới. Nhận xét kèm theo dựa trên train/holdout gốc, tách riêng khỏi chỉ số của dữ liệu vừa nhập.'}
