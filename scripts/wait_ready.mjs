const base=process.env.BASE_URL||'http://127.0.0.1:3000/api';
let ready=false;
for(let n=0;n<100;n++){
 try{const r=await fetch(base+'/health',{signal:AbortSignal.timeout(1000)});if(r.ok&&(await r.json()).status==='ok'){ready=true;break;}}catch{}
 await new Promise(resolve=>setTimeout(resolve,250));
}
if(!ready){console.error('Flask/Express chưa sẵn sàng.');process.exitCode=1;}else console.log('Flask/Express ready.');
