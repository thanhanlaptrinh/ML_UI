import assert from 'node:assert/strict';
const base=process.env.BASE_URL||'http://127.0.0.1:3000/api';
const get=async path=>{const r=await fetch(base+path);assert.equal(r.status,200);return r.json()};
const audit=await get('/audit');assert.equal(audit.models.length,5);assert.equal(audit.samples.length,30);assert.equal(audit.currency,'VND');
for(const model of audit.models){
 const sample=audit.samples[0];const response=await fetch(base+'/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:model.id,sample_id:sample.id})});
 assert.equal(response.status,200);const result=await response.json();assert.equal(result.actual,sample.actual);assert.ok(Math.abs(result.prediction-sample.predictions[model.id])<1);assert.ok(result.trace.verified);
 const calc=await get('/calculations/'+model.id);assert.equal(calc.rows.length,30);assert.ok(Math.abs(calc.metrics.mae-model.metrics.demo.mae)<1);
}
const csv=await fetch(base+'/export.csv');assert.equal(csv.status,200);assert.equal((await csv.text()).trim().split('\n').length,151);
const invalid=await fetch(base+'/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'bad',sample_id:1})});assert.equal(invalid.status,400);
console.log('PASS HTTP: five real pipelines, prices/metrics, decomposition, CSV 150 rows and invalid model.');
