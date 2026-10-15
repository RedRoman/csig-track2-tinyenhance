"""Actual CUDA output/RNG equivalence check for the fixed-caption cache."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path / 'runner.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Runner(str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original', type=Path, required=True)
    ap.add_argument('--cached', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    import torch
    import numpy as np
    torch.cuda.set_device(0)

    def state():
        ns = np.random.get_state()
        return dict(python=random.getstate(), numpy=(ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
                    cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(0).clone())

    def equal_state(x, y):
        return dict(python=x['python'] == y['python'], numpy=x['numpy'] == y['numpy'],
                    cpu=torch.equal(x['cpu'], y['cpu']), cuda=torch.equal(x['cuda'], y['cuda']))

    def quant(y, rounding):
        pixels=((y * 0.5 + 0.5) * 255).clamp(0,255)
        return (pixels.round() if rounding else pixels).to(torch.uint8)

    def sha(t):
        return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    began=time.time()
    original=load(a.original.resolve(), 'runner_original_cache_validation')
    initial=state()
    generator=torch.Generator(device='cuda:0').manual_seed(987654321)
    inputs=[torch.rand((1,3,h,w),device='cuda:0',dtype=torch.float16,generator=generator)*2-1
            for h,w in [(512,512),(768,1024),(512,512)]]
    baseline=[]
    with torch.no_grad():
        for x in inputs:
            y=original.infer(x);torch.cuda.synchronize()
            baseline.append((y.clone(),state()))
    original_protocol=original.protocol
    del original;torch.cuda.empty_cache()
    cached=load(a.cached.resolve(), 'runner_cached_cache_validation')
    initial_checks=equal_state(initial,state())
    report=dict(status='running',initial_rng_equal=initial_checks,checks=[],
                original_protocol=original_protocol,cached_protocol=cached.protocol,
                scope='three continuous forwards:512square,768x1024native,512square; original/cached both seed231-before-init')
    try:
        if not all(initial_checks.values()):
            raise RuntimeError('Cache initialization changed RNG')
        with torch.no_grad():
            for index,(x,(old,rng)) in enumerate(zip(inputs,baseline)):
                y=cached.infer(x);torch.cuda.synchronize()
                rng_checks=equal_state(rng,state())
                raw_equal=torch.equal(y,old)
                floor_equal=torch.equal(quant(y,False),quant(old,False))
                round_equal=torch.equal(quant(y,True),quant(old,True))
                check=dict(index=index,shape=list(x.shape),gpu_float_bitexact=raw_equal,
                           gpu_float_max_abs_delta=float((y-old).abs().max().item()),
                           uint8_floor_equal=floor_equal,uint8_round_equal=round_equal,
                           floor_sha256=sha(quant(y,False)),round_sha256=sha(quant(y,True)),rng_equal=rng_checks)
                report['checks'].append(check)
                if not raw_equal or not floor_equal or not round_equal or not all(rng_checks.values()):
                    raise RuntimeError('Cache output or forward RNG differs')
        report.update(status='passed',elapsed_s=time.time()-began)
    except Exception as error:
        report.update(status='failed',error=repr(error),elapsed_s=time.time()-began)
        raise
    finally:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'status':report['status'],'elapsed_s':report['elapsed_s'],'checks':report['checks']}),flush=True)


if __name__ == '__main__':
    main()
