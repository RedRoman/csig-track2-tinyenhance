"""Validate and benchmark isolated HYPIR cache variants on reserved GPU0."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default='/cache/independent_sr_cuda_20261013')
    a=ap.parse_args();r=a.root.resolve();code=r/'code/hypir_cuda_runner/cache_variants'
    out=r/'calibration/cache_variants_status.json'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    python=str(r/'env/bin/python')
    def status(stage,**fields):
        value=dict(stage=stage,physical_gpu=0,time=time.time(),**fields)
        temp=out.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(out)
        print(json.dumps(value),flush=True)
    def run(stage,command):
        status(stage,command=command)
        with (r/f'logs/cache_variants_{stage}.log').open('ab',buffering=0) as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (r/'device_locks/gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if uuid in apps:
            status('blocked_gpu0_busy');raise RuntimeError('GPU0 has an existing process')
        try:
            original=r/'submissions/hypir_s6000_mt200_ct175/model_dir'
            cached=r/'submissions/hypir_s6000_mt200_ct175_cached/model_dir'
            fused=r/'submissions/hypir_s6000_mt200_ct175_cached_fused/model_dir'
            validation=r/'calibration/cache_variants_validation.json'
            run('validation',[python,'-u',str(code/'validate_cached.py'),'--original',str(original),
                              '--cached',str(cached),'--out',str(validation)])
            if json.loads(validation.read_text())['status']!='passed':
                raise RuntimeError('Cache output/RNG validation failed; no cache timing')
            ref=json.loads((r/'calibration/g4_latency.json').read_text())['latency_512']['event_p10']
            summary={}
            for name,modeldir in [('cached',cached),('cached_fused',fused)]:
                result=r/f'calibration/hypir_s6000_ct175_{name}_latency.json'
                cmd=[python,'-u',str(r/'code/benchmark_cuda.py'),'--model-dir',str(modeldir),
                     '--out',str(result),'--ref-ms',str(ref),'--n','200','--warm','30','--device','0','--seed','231']
                for path in [r/'source_payload/adaptation/runs/ascend_balanced/step6000.pth',
                             r/'repos/HYPIR_cuda/HYPIR/enhancer/base.py',r/'repos/HYPIR_cuda/HYPIR/enhancer/sd2.py',
                             r/'repos/HYPIR_cuda/HYPIR/utils/common.py',
                             Path('/temp/zihaohe_stage/baselines/sd21b/unet/diffusion_pytorch_model.safetensors'),
                             Path('/temp/zihaohe_stage/baselines/sd21b/vae/diffusion_pytorch_model.safetensors'),
                             Path('/temp/zihaohe_stage/baselines/sd21b/text_encoder/model.safetensors')]:
                    cmd+=['--hash-file',str(path)]
                run(name,cmd)
                j=json.loads(result.read_text())
                if j['status']!='complete':raise RuntimeError('Benchmark did not complete')
                summary[name]=j['latency_512']
            status('complete',cache_equivalence='GPUfloat/uint8floor/round/RNG passed',
                   fused_quality='independent version; unverified',latency=summary)
        except Exception as error:
            status('failed',error=repr(error));raise


if __name__=='__main__':
    main()
