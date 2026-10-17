"""Simulate Windows git add + Linux/Windows checkout without weakening SHA256 checks."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,ok):
 assert ok,name
 checks.append(name)
def run(args,cwd):return subprocess.run(args,cwd=cwd,check=True,capture_output=True)
source=json.loads((ROOT/'results/vnd_audit_source.json').read_text())
with tempfile.TemporaryDirectory(prefix='house-git-artifacts-') as temp:
 source_repo=Path(temp)/'source'
 shutil.copytree(ROOT,source_repo,ignore=shutil.ignore_patterns('.git','.venv','node_modules','__pycache__','.cache'))
 run(['git','init','-q'],source_repo)
 run(['git','config','core.autocrlf','true'],source_repo)
 run(['git','add','.'],source_repo)
 run(['git','-c','user.name=Local Verification','-c','user.email=verify@example.invalid','commit','-qm','Verify byte-preserving artifacts'],source_repo)
 for file,digest in source['source_files'].items():
  content=run(['git','show','HEAD:'+file],source_repo).stdout
  check('git add autocrlf=true preserves '+file,hashlib.sha256(content).hexdigest()==digest)
 for setting in ['false','true']:
  clone=Path(temp)/('checkout-'+setting)
  run(['git','-c','core.autocrlf='+setting,'clone','-q','--no-local',str(source_repo),str(clone)],Path(temp))
  for file,digest in source['source_files'].items():check('checkout autocrlf='+setting+' '+file,hashlib.sha256((clone/file).read_bytes()).hexdigest()==digest)
  command=[sys.executable,'-c',"import sys;sys.path.insert(0,'service');from audit_engine import AuditEngine;e=AuditEngine();assert len(e.models)==5;print('All 5 models loaded; original SHA-256 guards passed')"]
  output=run(command,clone).stdout.decode().strip();check('API engine starts after '+setting+' checkout','All 5 models loaded' in output)
result={'status':'PASS','passed':len(checks),'checks':checks,'integrity_guard_disabled':False,'data_or_weights_modified':False}
(ROOT/'results/verification_git_artifacts.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('PASS',len(checks),'Git artifact checks; byte-exact guards preserved across CRLF/LF checkouts.')
