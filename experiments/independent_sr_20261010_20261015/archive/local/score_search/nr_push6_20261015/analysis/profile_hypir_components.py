"""Diagnostic CUDA-event component profiling. Default only prints a CPU plan.

Run with --execute on a root-selected idle GPU, one fresh process per model.
Instrumentation changes dispatch overhead: this is never an official benchmark,
full-quality result, or proof that a candidate preserves A7750 quality.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import importlib.util
import json
import os
import pickle
import random
import statistics
import time

R=Path('/cache/independent_sr_cuda_20261013')
WEIGHT='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--suite',choices=['smoke512','native7','native16','native41','native83'],default='smoke512')
    ap.add_argument('--warmup',type=int,default=30);ap.add_argument('--n',type=int,default=8)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    need(1<=a.n<=20 and 0<=a.warmup<=40,'Diagnostic limits exceeded')
    need(not a.out.exists(),'Immutable diagnostic output already exists')
    md=a.model_dir.resolve(strict=True);c=json.loads((md/'config.json').read_text())
    weight=Path(c['weight_path']);weight=weight if weight.is_absolute() else md/weight
    need(sha(weight)==WEIGHT,'Only unchanged real A7750 checkpoint permitted')
    weight_stat=weight.stat();weight_guard=(weight_stat.st_size,weight_stat.st_mtime_ns,weight_stat.st_ino)
    plan=dict(scope='diagnostic only; not benchmark or full quality',execute=a.execute,model_dir=str(md),suite=a.suite,warmup=a.warmup,n=a.n,out=str(a.out),checkpoint_sha256=WEIGHT,runner_sha256=sha(md/'runner.py'),helper_sha256=sha(__file__),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    print(json.dumps(plan),flush=True)
    if not a.execute:return
    import numpy as np
    import torch
    from PIL import Image
    need(torch.cuda.is_available(),'CUDA unavailable')
    spec=importlib.util.spec_from_file_location('runner',md/'runner.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    os.chdir(md);r=module.Runner(str(md));torch.cuda.synchronize(r.device)
    def rng():
        return dict(python=hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest(),numpy=hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=4)).hexdigest(),cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),cuda=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()])
    initial_rng=rng()
    if a.suite=='smoke512':
        v=torch.arange(3*512*512,device=r.device,dtype=torch.float32).reshape(1,3,512,512)
        x=(v.remainder(256)/255).half()*2-1;input_provenance='deterministic arange/remainder; no RNG input creation'
    else:
        p=R/'source_payload/assets/input'/('case'+a.suite.removeprefix('native')+'.jpg')
        with Image.open(p) as im:arr=np.asarray(im.convert('RGB'),dtype=np.float32)/255
        x=torch.from_numpy(arr).permute(2,0,1)[None].to(r.device).half()*2-1
        input_provenance=dict(path=str(p),sha256=sha(p))
    with torch.no_grad():
        for _ in range(a.warmup):y=r.infer(x)
    torch.cuda.synchronize(r.device)
    if a.warmup:del y
    after_warmup_rng=rng()
    # Wrap compiled CALLS after shape warmup; never trace event operations inside
    # compiled regions, change tensor arithmetic, or synchronize between stages.
    active=[];undo=[]
    def wrapped(fn,name):
        @functools.wraps(fn)
        def call(*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            t=time.perf_counter();start.record();value=fn(*args,**kwargs);end.record()
            active.append((name,start,end,(time.perf_counter()-t)*1000));return value
        return call
    def replace(obj,name,label):
        old=getattr(obj,name);setattr(obj,name,wrapped(old,label));undo.append((obj,name,old))
    import HYPIR.enhancer.base as base
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    original_tiler=base.make_tiled_fn
    def tiler(*args,**kwargs):
        fn=original_tiler(*args,**kwargs)
        return wrapped(fn,'tiles.'+str(kwargs.get('desc','unknown')))
    base.make_tiled_fn=tiler;undo.append((base,'make_tiled_fn',original_tiler))
    replace(r.model.vae,'encode','core.VAE.encode')
    replace(DiagonalGaussianDistribution,'sample','core.posterior.sample')
    replace(r.model.G,'forward','core.UNet.forward')
    replace(r.model.scheduler,'step','core.DDPM.step')
    replace(r.model,'forward_generator','generator.including_scale_UNet_DDPM')
    replace(r.model.vae,'decode','core.VAE.decode')
    replace(base,'wavelet_reconstruction','wavelet.full_reconstruction')
    samples=[];held=None
    try:
        with torch.no_grad():
            for i in range(a.n):
                active.clear();torch.cuda.synchronize(r.device);began=time.perf_counter()
                y=wrapped(r.infer,'inference.total')(x);torch.cuda.synchronize(r.device)
                wall=(time.perf_counter()-began)*1000;events={}
                for name,start,end,cpu_ms in active:
                    z=events.setdefault(name,dict(calls=0,GPU_stream_interval_ms=0.,CPU_dispatch_ms=0.))
                    z['calls']+=1;z['GPU_stream_interval_ms']+=start.elapsed_time(end);z['CPU_dispatch_ms']+=cpu_ms
                cpu=y.detach().cpu().clone()
                need(y.dtype==torch.float32 and y.device==x.device and y.shape==x.shape and bool(torch.isfinite(cpu).all()) and float(cpu.min())>=-1 and float(cpu.max())<=1,'Output contract differs')
                if held is not None:need(torch.equal(held[0].detach().cpu(),held[1]),'Retained output overwritten by subsequent inference')
                held=(y,cpu)
                # Inclusive groups overlap their child core timings. The
                # residual is stream elapsed overhead, not a sum of kernel time.
                residual={}
                for parent,children in {
                    'tiles.VAE encoding':['core.VAE.encode','core.posterior.sample'],
                    'tiles.Generator Forward':['generator.including_scale_UNet_DDPM'],
                    'tiles.VAE decoding':['core.VAE.decode'],
                    'generator.including_scale_UNet_DDPM':['core.UNet.forward','core.DDPM.step'],
                    'inference.total':['tiles.VAE encoding','tiles.Generator Forward','tiles.VAE decoding','wavelet.full_reconstruction'],
                }.items():
                    need(parent in events and all(k in events for k in children),'Expected profile scope absent')
                    residual[parent]=events[parent]['GPU_stream_interval_ms']-sum(events[k]['GPU_stream_interval_ms'] for k in children)
                row=dict(index=i,wall_selected_device_sync_ms=wall,inclusive_scopes=events,derived_stream_interval_residual_ms=residual,post_call_four_RNG_SHA256=rng(),output_float_SHA256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),finite_range_shape_contract_PASS=True)
                samples.append(row);print(json.dumps({'sample':i,'wall_ms':wall,'events':events}),flush=True)
    finally:
        for obj,name,old in reversed(undo):setattr(obj,name,old)
    names=samples[0]['inclusive_scopes'];medians={name:{k:statistics.median(s['inclusive_scopes'][name][k] for s in samples) for k in ['calls','GPU_stream_interval_ms','CPU_dispatch_ms']} for name in names}
    weight_stat=weight.stat();need((weight_stat.st_size,weight_stat.st_mtime_ns,weight_stat.st_ino)==weight_guard and sha(md/'runner.py')==plan['runner_sha256'],'Checkpoint/runner source changed')
    for name,digest in r.protocol['source_sha256'].items():need(sha(r.repo/name)==digest,'Original enhancer source changed')
    report=dict(**plan,status='complete',protocol=r.protocol,input_shape=list(x.shape),input_provenance=input_provenance,initial_four_RNG_SHA256=initial_rng,after_warmup_four_RNG_SHA256=after_warmup_rng,output_ownership_preserved=True,samples=samples,median_inclusive_scopes=medians,interpretation='Inclusive scope timings overlap child scopes. Events measure current-stream intervals including CPU enqueue gaps; compiled other-stream work may require joins. Wall synchronizes the entire selected device. Event instrumentation adds overhead. This is not an official latency or output-equivalence test.',original_code_files_unchanged=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'complete','output':str(a.out),'sha256':sha(a.out),'median_inclusive_scopes':medians}),flush=True)


if __name__=='__main__':main()
