"""Fresh serial single-GPU timing of G4, A7750, trained HYPIR and G4.

No training or scoring; rejects active GPU work before the batch. The two G4
measurements expose reference drift; the faster reference is selected before
looking at scores, giving the more conservative prediction for both models.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess, sys, time

R = Path('/cache/independent_sr_cuda_20261013')
GPU0 = 'GPU-823adbe3-cd4b-354f-3797-a941e3cabc9f'
BENCH_SHA = '3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
CM_SHA = ['3e422d0e8474532a7fc20838504c743c32e19b03770e440eaada6eee003d38f3',
          'd7e03918355aa9cb459c1d37f8ffafcee89856bb8b23cdf0346b5e440b3046cb']
MODELS = [R/'submissions/hypir_push7_a7750_trt_unet_vae_fp16/model_dir',
          R/'submissions/hypir_push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae/model_dir']

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def query(args): return subprocess.check_output(['nvidia-smi', *args], text=True).strip()
def snapshot():
    return dict(utc=now(), gpus=query(['--query-gpu=index,uuid,pci.bus_id,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw', '--format=csv,noheader']),
                processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader']))
def atomic(p,d):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n'); t.replace(p)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.out_dir.resolve()
    if not out.is_relative_to(R/'calibration') or out.exists(): raise RuntimeError('Use a new calibration directory')
    script=R/'code/benchmark_cuda.py'
    if sha(script)!=BENCH_SHA: raise RuntimeError('Original benchmark changed')
    jobs=[('g4_pre',R/'calibration/g4/model_dir',[])]
    for i,(label,md) in enumerate(zip(['reference','candidate'],MODELS)):
        cm=md.parent/'candidate_manifest.json'
        if sha(cm)!=CM_SHA[i]: raise RuntimeError('Actual registered candidate changed')
        c=json.loads(cm.read_text())
        extra=[Path(c['checkpoint']),cm]
        for x in c['actual_engines'].values(): extra.extend([Path(x['path']),Path(x['manifest'])])
        jobs.append((label,md,extra))
    jobs.append(('g4_post',R/'calibration/g4/model_dir',[]))
    if not args.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',out_dir=str(out),jobs=[x[0] for x in jobs],n=200,warm=30,seed=231,physical_GPU0=GPU0,reference_rule='min(G4_pre.event_p10,G4_post.event_p10)')))
        return
    initial=snapshot()
    if initial['processes']: raise RuntimeError('GPU work active; do not start timing')
    first=initial['gpus'].splitlines()[0].split(',')
    if first[0].strip()!='0' or first[1].strip()!=GPU0: raise RuntimeError('GPU identity changed')
    out.mkdir(parents=True)
    evidence=dict(schema='recovered_hypir_lastres_serial_cuda_v1',status='RUNNING',start_utc=now(),physical_GPU0_UUID=GPU0,
                  benchmark_sha256=BENCH_SHA,launcher_sha256=sha(__file__),n=200,warm=30,seed=231,
                  reference_rule='min(G4_pre.event_p10,G4_post.event_p10); fixed before measurements',
                  initial_snapshot=initial,jobs=[])
    receipt=out/'root_launch_evidence.json';atomic(receipt,evidence)
    for label,md,extra in jobs:
        before=snapshot()
        if before['processes']: raise RuntimeError('GPU work active before '+label)
        cmd=[sys.executable,str(script),'--model-dir',str(md),'--out',str(out/(label+'_latency.json')),
             '--n','200','--warm','30','--seed','231','--device','0']
        for p in extra:cmd.extend(['--hash-file',str(p)])
        row=dict(label=label,command=cmd,before=before,start_utc=now(),CUDA_VISIBLE_DEVICES='0',process_samples=[],interference=[])
        began=time.monotonic()
        with (out/(label+'.log')).open('x') as log:
            proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
            row['container_pid']=proc.pid
            while proc.poll() is None:
                sample=dict(utc=now(),processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']))
                row['process_samples'].append(sample)
                lines=sample['processes'].splitlines() if sample['processes'] else []
                if len(lines)>1 or any(x.split(',')[0].strip()!=GPU0 for x in lines):row['interference'].append(sample)
                time.sleep(2)
        # Let the completed child CUDA context leave NVML before the next job.
        after=snapshot()
        for _ in range(10):
            if not after['processes']:break
            time.sleep(1);after=snapshot()
        row.update(exit_code=proc.returncode,end_utc=now(),elapsed_seconds=time.monotonic()-began,after=after)
        if proc.returncode==0:row['latency_512']=json.loads((out/(label+'_latency.json')).read_text())['latency_512']
        evidence['jobs'].append(row);atomic(receipt,evidence)
        print(json.dumps(dict(label=label,exit_code=proc.returncode,latency=row.get('latency_512'),interference_count=len(row['interference']))),flush=True)
        if proc.returncode or row['interference'] or after['processes']: raise RuntimeError('Timing failed or interfered: '+label)
    evidence.update(status='COMPLETE_FOUR_FRESH_PROCESSES',end_utc=now(),final_snapshot=snapshot())
    atomic(receipt,evidence)

if __name__=='__main__': main()
