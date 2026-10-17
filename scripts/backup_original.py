"""One-time immutable baseline snapshot. Existing manifests are verified, never replaced."""
import hashlib,json,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def backup():
 manifest=ROOT/'models/_backup_original/manifest.json'
 if manifest.exists():
  r=json.loads(manifest.read_text())
  for f in r['files']:assert sha(ROOT/f['backup'])==f['sha256'],f['backup']
  return r
 models=[ROOT/'models/best_model.pkl',ROOT/'models/feature_defaults.json',*sorted((ROOT/'models/vnd_supplement').glob('*.pkl'))]
 results=[p for p in sorted((ROOT/'results').iterdir()) if p.is_file()]
 entries=[]
 for p in models+results:
  dest=(ROOT/'models/_backup_original'/p.relative_to(ROOT/'models')) if p in models else ROOT/'results/baseline_original'/p.name
  if dest.exists():raise FileExistsError(dest)
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  assert sha(dest)==sha(p)
  entries.append({'original':str(p.relative_to(ROOT)),'backup':str(dest.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size})
 snap=ROOT/'results/baseline_original/source_before_round1.zip'
 with zipfile.ZipFile(snap,'w',zipfile.ZIP_DEFLATED) as z:
  for folder in ['service','scripts','notebooks_or_src_py','config','public']:
   for p in sorted((ROOT/folder).rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
  for f in ['server.js','README.md','README_DYNAMIC.md','README_VND.md','HUONG_DAN_THANH.md']:z.write(ROOT/f,f)
 r={'created_at':datetime.now(timezone.utc).isoformat(),'files':entries,'source_snapshot_sha256':sha(snap),'data_hashes':{f:sha(ROOT/f) for f in ['data/train.csv','data/test.csv']}}
 manifest.write_text(json.dumps(r,indent=2),encoding='utf8');(ROOT/'results/baseline_original/manifest.json').write_text(json.dumps(r,indent=2),encoding='utf8')
 return r
if __name__=='__main__':print('BACKUP VERIFIED:',len(backup()['files']),'original artifacts')
