"""Reserved-GPU0 compile attempt with a ten-minute subprocess-group budget."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default='/cache/independent_sr_cuda_20261013')
    a=ap.parse_args();r=a.root.resolve();code=r/'code/hypir_cuda_runner/compile_variant'
    out=r/'calibration/compile_variant_status.json'
    def status(stage,**fields):
        j=dict(stage=stage,physical_gpu=0,time=time.time(),**fields)
        t=out.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2));t.replace(out);print(json.dumps(j),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
             TORCHINDUCTOR_CACHE_DIR=str(r/'compile_cache/hypir_tiles'),
             TRITON_CACHE_DIR=str(r/'compile_cache/triton'))
    python=str(r/'env/bin/python')
    baseline=r/'calibration/hypir_s8000_ct200_latency.json'
    pid=json.loads((r/'calibration/hypir_s8000_ct200_latency_launch.json').read_text())['pid']
    while not baseline.exists() or Path(f'/proc/{pid}').exists():
        status('waiting_baseline8k',baseline_pid=pid);time.sleep(5)
    if json.loads(baseline.read_text())['status']!='complete':
        status('baseline_failed');raise RuntimeError('Baseline8k failed')
    with (r/'device_locks/gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if uuid in apps:
            status('blocked_gpu0_busy');raise RuntimeError('GPU0 has an existing process')
        deadline=time.monotonic()+600
        def run(stage,cmd):
            remaining=deadline-time.monotonic()
            if remaining<10:raise TimeoutError('Ten-minute compile budget exhausted')
            status(stage,budget_remaining_s=remaining)
            with (r/f'logs/compile_variant_{stage}.log').open('ab',buffering=0) as log:
                p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    result=p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    raise TimeoutError('Own compile subprocess group terminated at ten-minute limit')
                if result:raise RuntimeError(f'{stage} exited {result}')
        try:
            model=r/'submissions/hypir_s6000_mt200_ct175_cached_fused_compiled/model_dir'
            eager=r/'submissions/hypir_s6000_mt200_ct175_cached_fused/model_dir'
            validation=r/'calibration/compile_variant_validation.json'
            run('validation',[python,'-u',str(code/'validate_compile.py'),'--original',str(eager),
                              '--cached',str(model),'--out',str(validation)])
            v=json.loads(validation.read_text())
            if v['status']!='compare_complete':raise RuntimeError('Compiled comparison did not complete')
            ref=json.loads((r/'calibration/g4_latency.json').read_text())['latency_512']['event_p10']
            result=r/'calibration/hypir_s6000_ct175_cached_fused_compiled_latency.json'
            cmd=[python,'-u',str(r/'code/benchmark_cuda.py'),'--model-dir',str(model),'--out',str(result),
                 '--ref-ms',str(ref),'--warm','30','--n','200','--device','0','--seed','231']
            for p in [r/'source_payload/adaptation/runs/ascend_balanced/step6000.pth',
                      r/'repos/HYPIR_cuda/HYPIR/enhancer/base.py',r/'repos/HYPIR_cuda/HYPIR/enhancer/sd2.py',
                      r/'repos/HYPIR_cuda/HYPIR/utils/common.py']:
                cmd+=['--hash-file',str(p)]
            run('benchmark',cmd)
            j=json.loads(result.read_text())
            if j['status']!='complete':raise RuntimeError('Compiled benchmark failed')
            status('complete',latency=j['latency_512'],gpu_float_bitexact=v['float_bitexact'],
                   uint8_round_equal=v['round_uint8_equal'],rng_identical=v['rng_identical'],
                   quality='independent compiled/fused version; full score unverified')
        except Exception as e:
            status('timeout' if isinstance(e,TimeoutError) else 'failed',error=repr(e));raise


if __name__=='__main__':
    main()
