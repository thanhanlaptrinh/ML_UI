const express=require('express');
const path=require('node:path');
const app=express();
const port=Number(process.env.PORT||3000);
const service=(process.env.PREDICT_SERVICE_URL||'http://127.0.0.1:5001').replace(/\/$/,'');
app.disable('x-powered-by');app.use(express.json({limit:'4mb'}));
app.use(express.static(path.join(__dirname,'public')));
async function proxy(req,res,endpoint){
 try{
  const response=await fetch(service+endpoint,{method:req.method,headers:req.method==='POST'?{'Content-Type':'application/json'}:{},body:req.method==='POST'?JSON.stringify(req.body):undefined,signal:AbortSignal.timeout(60000)});
  res.status(response.status);res.set('Content-Type',response.headers.get('content-type')||'application/json');
  if(response.headers.get('content-disposition'))res.set('Content-Disposition',response.headers.get('content-disposition'));
  res.send(await response.text());
 }catch(err){res.status(503).json({error:'Chưa kết nối Flask. Chạy python service/predict_service.py ở cửa sổ thứ hai.',detail:err.message});}
}
for(const route of ['audit','health','proof','export.csv','schema'])app.get('/api/'+route,(req,res)=>proxy(req,res,'/'+route));
app.get('/api/calculations/:model',(req,res)=>proxy(req,res,'/calculations/'+encodeURIComponent(req.params.model)));
app.get('/api/dataset/:kind',(req,res)=>proxy(req,res,'/dataset/'+encodeURIComponent(req.params.kind)));
app.get('/api/record/:id',(req,res)=>proxy(req,res,'/record/'+encodeURIComponent(req.params.id)));
app.post('/api/evaluate',(req,res)=>proxy(req,res,'/evaluate'));
app.post('/api/training-report',(req,res)=>proxy(req,res,'/training-report'));
app.post('/api/predict',(req,res)=>proxy(req,res,'/predict'));
app.post('/predict',(req,res)=>proxy(req,res,'/predict'));
app.listen(port,()=>console.log(`HomeValue Lab VND: http://localhost:${port}`));
