"""Root-owned same-idle-A100 serial timing; no quality reuse."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
R=Path('/cache/independent_sr_cuda_20261013')
D=R/'calibration/push7_teacher_formal_20261015T0604'
RAW=R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
ENG=R/'backends/push7_trt'
jobs=[
 ('g4',R/'calibration/g4/model_dir',[]),
 ('push6',R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir',[RAW]),
 ('trtvae',R/'submissions/hypir_push7_a7750_trtvae_fp16/model_dir',[RAW,ENG/'encoder_fp16/component.engine',ENG/'decoder_fp16/component.engine']),
 ('trtunetvae',R/'submissions/hypir_push7_a7750_trt_unet_vae_fp16/model_dir',[RAW,ENG/'encoder_fp16/component.engine',ENG/'decoder_fp16/component.engine',ENG/'unet_fp16_A7750/component.engine']),
]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True)
def snapshot():
 return dict(utc=now(),gpus=query(['--query-gpu=index,uuid,pci.bus_id,name,utilization.gpu,memory.used','--format=csv,noheader']),processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']))
def save(x):
 p=D/'root_launch_evidence.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
if D.exists():raise RuntimeError('Use new immutable calibration directory')
D.mkdir(parents=True)
script=R/'code/benchmark_cuda.py'
digest=hashlib.sha256(script.read_bytes()).hexdigest()
if digest!='3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995':raise RuntimeError('Original benchmark changed')
e=dict(schema='push7_root_serial_idle_gpu0_v1',start_utc=now(),CUDA_VISIBLE_DEVICES='0',benchmark_sha256=digest,launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n=200,warm=30,seed=231,device=0,jobs=[],quality_reuse_claim=False,initial_snapshot=snapshot())
if e['initial_snapshot']['processes'].strip():raise RuntimeError('GPU work still active before formal serial run')
save(e)
for label,md,hashes in jobs:
 before=snapshot()
 if before['processes'].strip():raise RuntimeError('GPU work still active before '+label)
 cmd=[sys.executable,str(script),'--model-dir',str(md),'--out',str(D/(label+'_latency.json')),'--n','200','--warm','30','--seed','231','--device','0']
 for p in hashes:cmd.extend(['--hash-file',str(p)])
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
 row=dict(label=label,command=cmd,CUDA_VISIBLE_DEVICES='0',before=before,start_utc=now())
 started=time.monotonic()
 with (D/(label+'.log')).open('x') as stream:
  proc=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
 row.update(exit_code=proc.returncode,end_utc=now(),elapsed_seconds=time.monotonic()-started,after=snapshot())
 if proc.returncode==0:
  result=json.loads((D/(label+'_latency.json')).read_text());row['latency_512']=result['latency_512']
 e['jobs'].append(row);save(e)
 print(json.dumps(dict(label=label,exit_code=proc.returncode,latency=row.get('latency_512'))),flush=True)
 if proc.returncode!=0:raise RuntimeError('Benchmark failed: '+label)
e.update(status='COMPLETE_FOUR_FRESH_PROCESSES',end_utc=now(),final_snapshot=snapshot());save(e)

