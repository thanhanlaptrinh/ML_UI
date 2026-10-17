"""Execute mandatory checks against Flask/Express in one network namespace; retain actual output."""
import sys,subprocess,time,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
commands=[[sys.executable,'scripts/verify_vnd.py'],[sys.executable,'scripts/verify_dynamic.py'],[sys.executable,'scripts/verify_training_report.py'],[sys.executable,'scripts/verify_round1.py'],['node','scripts/smoke_http.mjs'],['node','scripts/smoke_dynamic_http.mjs']]
results=[];processes=[]
with (ROOT/'results/round1_servers.log').open('w') as log:
 try:
  processes=[subprocess.Popen([sys.executable,'service/predict_service.py'],cwd=ROOT,stdout=log,stderr=log),subprocess.Popen(['node','server.js'],cwd=ROOT,stdout=log,stderr=log)]
  ready=False
  for _ in range(120):
   try:
    with urllib.request.urlopen('http://127.0.0.1:3000/api/health',timeout=1) as response:
     if response.status==200:ready=True;break
   except Exception:pass
   time.sleep(.25)
  if not ready:raise RuntimeError('Flask/Express not ready; inspect results/round1_servers.log')
  for command in commands:
   t=time.perf_counter();p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
   r={'command':['python' if v==sys.executable else v for v in command],'exit_code':p.returncode,'status':'PASS' if p.returncode==0 else 'FAIL','stdout':p.stdout,'stderr':p.stderr,'seconds':time.perf_counter()-t};results.append(r)
   print('$',' '.join(r['command']),flush=True);print(p.stdout,end='',flush=True)
   if p.stderr:print(p.stderr,file=sys.stderr,flush=True)
 finally:
  for p in processes:
   p.terminate()
  for p in processes:
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill()
result={'status':'PASS' if len(results)==len(commands) and all(r['exit_code']==0 for r in results) else 'FAIL','results':results}
(ROOT/'results/round1_test_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('ROUND1 CHECKS:',result['status'])
sys.exit(0 if result['status']=='PASS' else 1)
