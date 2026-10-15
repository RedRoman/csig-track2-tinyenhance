"""Immediate CPU-only bounded watcher/auditor for Push4 pilot completions with exact stage5 marker binding.
No CUDA initialization, rendering, metric reruns, scheduler, or original-file edits.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, re, shutil, subprocess, time
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
RUNS={'a':'cuda_nr_stage5_s7250_teacher6500_continue500','b':'cuda_nr_stage5_s7250_teacher6500_m25_500','c':'cuda_nr_stage5_s7250_teacher6500_m25_r10_500'}
def stat_token(p):
 st=Path(p).stat();return st.st_size,st.st_mtime_ns,st.st_ino
def marker_guard(label,d):
 match=re.fullmatch(r'push4_([abc])(7500|7750)_compiled_pilot',label);assert match,label
 branch,step=match.group(1),int(match.group(2));local=step-7250
 rp=js(d/'render_protocol.json')['protocol'];raw=Path(rp['runner']['weight_path']).resolve()
 assert raw==ROOT/'adaptation/runs'/RUNS[branch]/f'step{step}.pth'
 marker=raw.parent/f'checkpoint_step{step}.json';m=js(marker)
 assert (m['local_step'],m['global_step'],m['optimizer_step'])==(local,step,1250+local)
 assert m['raw_checkpoint_sha256']==rp['runner']['checkpoint_sha256']
 registration=Path(rp['model_dir']).parent/'registration.json';reg=js(registration)
 assert reg['schema']=='nr_push4_candidate_registration_v1' and reg['checkpoint']['sha256']==m['raw_checkpoint_sha256']
 assert Path(reg['checkpoint']['path']).resolve()==raw
 return marker,m,raw,stat_token(raw),sha(marker)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',action='append',required=True);p.add_argument('--minutes',type=float,default=20);p.add_argument('--poll',type=float,default=25);a=p.parse_args();assert 0<a.minutes<=30 and 15<=a.poll<=60;assert len(a.label)==len(set(a.label));env=dict(os.environ,CUDA_VISIBLE_DEVICES='');end=time.monotonic()+a.minutes*60;out=ROOT/'audits/push4/evaluations';out.mkdir(parents=True,exist_ok=True);done={};last=None
 while len(done)<len(a.label) and time.monotonic()<end:
  status={}
  for label in a.label:
   d=ROOT/'nr_push4_pilot_evaluations'/label
   r=js(d/'scores/result.json') if (d/'scores/result.json').exists() else {}
   s=js(d/'render_status.json') if (d/'render_status.json').exists() else {}
   status[label]={'score_status':r.get('status'),'scored':r.get('completed',0),'render_status':s.get('status'),'rendered':s.get('completed',0)}
   if label in done:continue
   if r.get('status')=='complete' and r.get('completed')==23 and s.get('status')=='complete' and s.get('completed')==23:
    marker,marker_value,raw,raw_token,marker_sha=marker_guard(label,d)
    dest=out/label;dest.mkdir(parents=True,exist_ok=True)
    command=[str(ROOT/'env/bin/python'),str(ROOT/'code/push4/audit_push4_evaluation_cpu.py'),label,'--evals-root',str(ROOT/'nr_push4_pilot_evaluations')]
    with (dest/'audit.json.tmp').open('w') as stdout,(dest/'audit.stderr.log').open('w') as stderr:
     result=subprocess.run(command,env=env,stdout=stdout,stderr=stderr,timeout=max(1,min(120,end-time.monotonic())))
    if result.returncode:
     print(json.dumps({'label':label,'status':'AUDIT_FAILED','stderr':(dest/'audit.stderr.log').read_text()}),flush=True);return 1
    (dest/'audit.json.tmp').replace(dest/'audit.json');audit=js(dest/'audit.json');assert audit['actual_checkpoint_sha256']==marker_value['raw_checkpoint_sha256'];assert stat_token(raw)==raw_token and sha(marker)==marker_sha
    (dest/'proof').mkdir(parents=True,exist_ok=True);shutil.copyfile(marker,dest/'proof/checkpoint_marker.json');manifest=[{'path':'checkpoint_marker.json','source':str(marker),'bytes':marker.stat().st_size,'sha256':marker_sha}]
    for name in SMALL:
     src=d/name;target=dest/'proof'/name
     assert src.is_file() and src.stat().st_size<4*1024*1024,(name,'unexpected proof file size')
     target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);assert sha(src)==sha(target)
     manifest.append({'path':name,'source':str(src),'bytes':src.stat().st_size,'sha256':sha(target)})
    registration=Path(js(d/'render_protocol.json')['protocol']['model_dir']).parent/'registration.json'
    if registration.exists():
     target=dest/'proof/registration.json';shutil.copyfile(registration,target);manifest.append({'path':'registration.json','source':str(registration),'bytes':registration.stat().st_size,'sha256':sha(target)})
    atomic(dest/'proof_manifest.json',{'label':label,'scope':'fixed20+val3; not full score','auditor_sha256':sha(ROOT/'code/push4/audit_push4_evaluation_cpu.py'),'monitor_sha256':sha(__file__),'actual_stage5_checkpoint_marker_sha256':marker_sha,'global_step':marker_value['global_step'],'stage_local_step':marker_value['local_step'],'optimizer_step':marker_value['optimizer_step'],'actual_raw_checkpoint_matches_completed_marker':True,'files':manifest})
    summary={k:audit[k] for k in ['label','status','image_rows','scalar_metrics','NR','FR','S_subset','actual_checkpoint_sha256','reference_comparison']};atomic(dest/'summary.json',summary);done[label]=summary
    print(json.dumps({'event':'PILOT_AUDIT_PASS',**summary},ensure_ascii=False),flush=True)
  snapshot={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete' if len(done)==len(a.label) else 'watching','completed_audits':done,'pilot_status':status,'cpu_only':True}
  atomic(out/('monitor_'+str(len(a.label))+'pilots_status.json'),snapshot)
  if status!=last:print(json.dumps({'event':'STATUS','status':status}),flush=True);last=status
  if len(done)<len(a.label):time.sleep(min(a.poll,max(0,end-time.monotonic())))
 print(json.dumps({'event':'WATCH_END','status':'complete' if len(done)==len(a.label) else 'timeout','completed_labels':list(done)}),flush=True);return 0 if len(done)==len(a.label) else 2
if __name__=='__main__':raise SystemExit(main())
