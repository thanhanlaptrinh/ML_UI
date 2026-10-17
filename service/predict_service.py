"""VND audit API. Inference only: does not train or overwrite models."""
import csv
import io
from flask import Flask, jsonify, request, Response
from audit_engine import AuditEngine, MODEL_IDS

app=Flask(__name__)
engine=AuditEngine()
from flexible_engine import FlexibleEngine
flexible=FlexibleEngine(engine)
app.config['MAX_CONTENT_LENGTH']=4*1024*1024

@app.post('/training-report')
def training_report():
    from training_report import build_training_report
    return jsonify(build_training_report(engine))

@app.get('/schema')
def schema():
    return jsonify(flexible.schema())

@app.get('/dataset/<kind>')
def dataset(kind):
    try:return jsonify(flexible.dataset(kind))
    except ValueError as exc:return jsonify({'error':str(exc)}),400

@app.get('/record/<value>')
def record(value):
    try:return jsonify(flexible.lookup(value))
    except ValueError as exc:return jsonify({'error':str(exc)}),400

@app.post('/evaluate')
def evaluate():
    try:return jsonify(flexible.evaluate(request.get_json(silent=True)))
    except (ValueError,TypeError,KeyError) as exc:return jsonify({'error':str(exc)}),400


@app.get('/health')
def health():
    return jsonify({'status':'ok','models':5,'currency':'VND'})

@app.get('/audit')
def audit():
    return jsonify(engine.overview())

@app.get('/calculations/<slug>')
def calculations(slug):
    if slug not in MODEL_IDS:return jsonify({'error':'Mô hình không hợp lệ.'}),400
    return jsonify(engine.calculations(slug))

@app.post('/predict')
def predict():
    data=request.get_json(silent=True)
    if not isinstance(data,dict):return jsonify({'error':'Cần gửi JSON hợp lệ.'}),400
    try:
        return jsonify(engine.predict(data.get('model'),data.get('sample_id'),data.get('changes',{})))
    except (ValueError,TypeError,KeyError) as exc:
        return jsonify({'error':str(exc)}),400

@app.get('/proof')
def proof():
    return jsonify({'overview':engine.overview(),'calculations':{slug:engine.calculations(slug) for slug in MODEL_IDS},
                    'source_report':engine.report})

@app.get('/export.csv')
def export():
    buffer=io.StringIO();writer=csv.writer(buffer)
    writer.writerow(['model','source_Id','actual_VND','predicted_VND','residual_VND','absolute_error_VND','squared_error_VND2','APE_percent','USD_TO_VND'])
    from audit_engine import RATE
    for slug in MODEL_IDS:
        for r in engine.calculations(slug)['rows']:
            writer.writerow([slug,r['id'],r['actual'],r['prediction'],r['error'],r['absolute_error'],r['squared_error'],r['percentage_error'],RATE])
    return Response('\ufeff'+buffer.getvalue(),mimetype='text/csv; charset=utf-8',headers={'Content-Disposition':'attachment; filename=doi_chieu_30_can_VND.csv'})

if __name__=='__main__':app.run(host='127.0.0.1',port=5001,debug=False)
