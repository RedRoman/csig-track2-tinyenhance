from pathlib import Path
import datetime, hashlib, json, os, signal, subprocess, time
R=Path('/cache/independent_sr_cuda_20261013');D=R/'nr_push3_pilot_evaluations/preparation/compile_reduce_overhead_probe';D.mkdir(parents=True,exist_ok=True);P=R/'env/bin/python';ref=R/'submissions/hypir_cuda_nr_stage3_anchor6250_m15_s6500_compiled/model_dir';dst=R/'submissions/hypir_push3_6500_reduceoverhead_probe/model_dir';assert not dst.exists();dst.mkdir(parents=True)
source=(ref/'runner.py').read_text();assert source.count("mode='default', dynamic=False")==3;source=source.replace("mode='default', dynamic=False","mode='reduce-overhead', dynamic=False").replace('torch.compile default dynamicFalse','torch.compile reduce-overhead dynamicFalse');(dst/'runner.py').write_text(source);c=json.loads((ref/'config.json').read_text());c['label']='push3_6500_reduceoverhead_probe';(dst/'config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
start=time.monotonic();deadline=start+155;env=dict(os.environ,CUDA_VISIBLE_DEVICES='3');records=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=[('smoke_reference',[str(P),str(R/'code/push3/probe_equivalence.py'),'--model-dir',str(ref),'--out-dir',str(D),'--reference']),('smoke_candidate',[str(P),str(R/'code/push3/probe_equivalence.py'),'--model-dir',str(dst),'--out-dir',str(D)]),('benchmark_reference',[str(P),str(R/'code/benchmark_cuda.py'),'--model-dir',str(ref),'--out',str(D/'reference_latency.json'),'--warm','30','--n','200','--hash-file',c['weight_path']]),('benchmark_candidate',[str(P),str(R/'code/benchmark_cuda.py'),'--model-dir',str(dst),'--out',str(D/'candidate_latency.json'),'--warm','30','--n','200','--hash-file',c['weight_path']])]
result={'schema':'bounded_reduce_overhead_probe_v1','gpu_physical':3,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_sha256':sha(c['weight_path']),'reference_runner_sha256':sha(ref/'runner.py'),'candidate_runner_sha256':sha(dst/'runner.py'),'benchmark_script_sha256':sha(R/'code/benchmark_cuda.py'),'steps':records,'candidate_model_dir':str(dst),'status':'running'}
try:
 for name,cmd in plan:
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('155second total probe budget exhausted')
  print('START',name,'remaining',remaining,flush=True)
  with (D/(name+'.log')).open('w') as out:
   proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   try:code=proc.wait(timeout=remaining)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise TimeoutError(name+' exceeded bounded time budget; only own process group stopped')
  records.append({'name':name,'exit_code':code,'elapsed_s':time.monotonic()-start});print('END',name,code,flush=True)
  if code:raise RuntimeError(name+' failed')
 result['status']='complete'
except Exception as e:result.update(status='failed',error_type=type(e).__name__,error=str(e))
result['elapsed_s']=time.monotonic()-start;result['gpu3_released']=True
for name in ['reference_smoke','candidate_smoke','reference_latency','candidate_latency']:
 path=D/(name+'.json')
 if path.exists():result[name]=json.loads(path.read_text())
(D/'probe_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':result['status'],'error':result.get('error'),'elapsed_s':result['elapsed_s'],'gpu3_released':True,'out':str(D/'probe_result.json')}),flush=True)
