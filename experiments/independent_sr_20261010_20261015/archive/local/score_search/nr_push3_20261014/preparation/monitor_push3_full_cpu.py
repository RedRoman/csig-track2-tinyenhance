"""Immediate CPU-only bounded watcher/auditor for this round's full103 completions.
No CUDA initialization, rendering, metric reruns, scheduler, or original-file edits.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, subprocess, time
ROOT=Path('/cache/independent_sr_cuda_20261013')
SMALL=['render_protocol.json','render_manifest.json','render_status.json','render_progress.pt','scores/protocol.json','scores/result.json','scores/status.json','scores/measurements.jsonl','scores/rng_checkpoint.pt']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',action='append',required=True);p.add_argument('--minutes',type=float,default=30);p.add_argument('--poll',type=float,default=25);a=p.parse_args();env=dict(os.environ,CUDA_VISIBLE_DEVICES='');end=time.monotonic()+a.minutes*60;out=ROOT/'audits/push3/evaluations';out.mkdir(parents=True,exist_ok=True);done={};last=None
 while len(done)<len(a.label) and time.monotonic()<end:
  status={}
  for label in a.label:
   d=ROOT/'nr_push3_full_evaluations'/label
   r=js(d/'scores/result.json') if (d/'scores/result.json').exists() else {}
   s=js(d/'render_status.json') if (d/'render_status.json').exists() else {}
   status[label]={'score_status':r.get('status'),'scored':r.get('completed',0),'render_status':s.get('status'),'rendered':s.get('completed',0)}
   if label in done:continue
   if r.get('status')=='complete' and r.get('completed')==103 and s.get('status')=='complete' and s.get('completed')==103:
    dest=out/label;dest.mkdir(parents=True,exist_ok=True)
    command=[str(ROOT/'env/bin/python'),str(ROOT/'code/push3/audit_push3_evaluation_cpu.py'),label,'--evals-root',str(ROOT/'nr_push3_full_evaluations')]
    with (dest/'audit.json.tmp').open('w') as stdout,(dest/'audit.stderr.log').open('w') as stderr:
     result=subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
    if result.returncode:
     print(json.dumps({'label':label,'status':'AUDIT_FAILED','stderr':(dest/'audit.stderr.log').read_text()}),flush=True);return 1
    (dest/'audit.json.tmp').replace(dest/'audit.json');audit=js(dest/'audit.json');manifest=[]
    for name in SMALL:
     src=d/name;target=dest/'proof'/name
     assert src.is_file() and src.stat().st_size<4*1024*1024,(name,'unexpected proof file size')
     target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);assert sha(src)==sha(target)
     manifest.append({'path':name,'source':str(src),'bytes':src.stat().st_size,'sha256':sha(target)})
    registration=Path(js(d/'render_protocol.json')['protocol']['model_dir']).parent/'registration.json'
    if registration.exists():
     target=dest/'proof/registration.json';shutil.copyfile(registration,target);manifest.append({'path':'registration.json','source':str(registration),'bytes':registration.stat().st_size,'sha256':sha(target)})
    atomic(dest/'proof_manifest.json',{'label':label,'scope':'full100test+val3; S_proxy, not official leaderboard score','auditor_sha256':sha(ROOT/'code/push3/audit_push3_evaluation_cpu.py'),'monitor_sha256':sha(__file__),'files':manifest})
    summary={k:audit[k] for k in ['label','status','image_rows','scalar_metrics','NR','FR','S_proxy','actual_checkpoint_sha256','reference_comparison']};atomic(dest/'summary.json',summary);done[label]=summary
    print(json.dumps({'event':'FULL_AUDIT_PASS',**summary},ensure_ascii=False),flush=True)
  snapshot={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete' if len(done)==len(a.label) else 'watching','completed_audits':done,'full_status':status,'cpu_only':True}
  atomic(out/('monitor_'+str(len(a.label))+'full_status.json'),snapshot)
  if status!=last:print(json.dumps({'event':'STATUS','status':status}),flush=True);last=status
  if len(done)<len(a.label):time.sleep(min(a.poll,max(0,end-time.monotonic())))
 print(json.dumps({'event':'WATCH_END','status':'complete' if len(done)==len(a.label) else 'timeout','completed_labels':list(done)}),flush=True);return 0 if len(done)==len(a.label) else 2
if __name__=='__main__':raise SystemExit(main())
