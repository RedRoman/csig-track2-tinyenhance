"""Fixed-case image/four-RNG/ownership diagnostic; never an IQA or latency claim.

Separate fresh root-scheduled GPU processes for reference and each candidate.
Optional --reference-dir reuses old initial RNG/PTs read-only for the SAME suite
and order. New reference 16/83 is its own fresh sequence, never spliced into 7/41.
All output goes to a new directory below R/analysis/push6, never an old archive.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import io
import json
import os
import pickle
import random

R=Path('/cache/independent_sr_cuda_20261013')
PARENT=R/'submissions/hypir_push4_a7750_compiled/model_dir'
PARENT_SOURCE='396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae'
WEIGHT='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--phase',choices=['reference','candidate'],required=True)
    p.add_argument('--suite',choices=['native7_41','native16_83','native7_16_41_83'],required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path);a=p.parse_args()
    d=a.out_dir.resolve();model=a.model_dir.resolve(strict=True)
    need(d.is_relative_to(R/'analysis/push6'),'Use a new push6 diagnostic output directory')
    refdir=a.reference_dir.resolve(strict=True) if a.reference_dir else d
    need(not a.reference_dir or (a.phase=='candidate' and refdir!=d and not d.is_relative_to(refdir)),'Reused references must remain read-only')
    cfg=json.loads((model/'config.json').read_text());weight=Path(cfg['weight_path']);weight=weight if weight.is_absolute() else model/weight
    need(sha(weight)==WEIGHT,'Not the unchanged A7750 checkpoint')
    if a.phase=='reference':need(model==PARENT and sha(model/'runner.py')==PARENT_SOURCE,'Wrong A7750 reference runner')
    else:
        m=json.loads((model.parent/'candidate_manifest.json').read_text())
        need(sha(model/'runner.py')==m['runner_sha256'] and sha(model/'config.json')==m['config_sha256'] and m['checkpoint_sha256']==WEIGHT,'Unbound candidate source/config')
    keys={'native7_41':[7,41],'native16_83':[16,83],'native7_16_41_83':[7,16,41,83]}[a.suite]
    name=a.suite+'_'+a.phase+('_'+model.parent.name if a.phase=='candidate' else '')
    result=d/(name+'.json');need(not result.exists(),'Immutable diagnostic output exists')
    plan=dict(execute=a.execute,phase=a.phase,suite=a.suite,cases=keys,model_dir=str(model),out_dir=str(d),reference_dir=str(refdir),runner_sha256=sha(model/'runner.py'),checkpoint_sha256=WEIGHT,helper_sha256=sha(__file__))
    print(json.dumps(plan),flush=True)
    if not a.execute:return
    import numpy as np
    import torch
    from PIL import Image
    def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    def rh(s):return dict(python=hashlib.sha256(pickle.dumps(s['python'],protocol=4)).hexdigest(),numpy=hashlib.sha256(pickle.dumps(s['numpy'],protocol=4)).hexdigest(),cpu=hashlib.sha256(s['cpu'].cpu().numpy().tobytes()).hexdigest(),cuda=[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in s['cuda']])
    def restore(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])
    def png(cpu):
        pixels=((cpu[0].permute(1,2,0).float().clamp(-1,1).numpy()*.5+.5)*255).round().astype(np.uint8)
        b=io.BytesIO();Image.fromarray(pixels).save(b,format='PNG',compress_level=1);return b.getvalue()
    def save_png(path,cpu):
        data=png(cpu);digest=hashlib.sha256(data).hexdigest()
        if path.exists():need(sha(path)==digest,'Existing new diagnostic PNG differs')
        else:
            with path.open('xb') as f:f.write(data)
        return dict(path=str(path),sha256=digest,bytes=len(data))
    d.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('runner',model/'runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    os.chdir(model);r=mod.Runner(str(model));torch.cuda.synchronize(r.device);state=rng()
    initial=refdir/(a.suite+'_initial_rng.pt');refs={};source_heads={}
    if a.phase=='reference':
        need(not initial.exists(),'Reference initial RNG already exists');torch.save(state,initial)
    else:
        need(initial.is_file(),'Missing same-suite reference RNG');source_heads[str(initial)]=sha(initial)
        saved=torch.load(initial,map_location='cpu',weights_only=False)
        need(rh(saved)==rh(state),'Constructor four-RNG differs');restore(saved)
    rows=[];held=[]
    with torch.no_grad():
        for case in keys:
            src=R/'source_payload/assets/input'/f'case{case}.jpg'
            with Image.open(src) as im:arr=np.asarray(im.convert('RGB'),dtype=np.float32)/255.0
            x=torch.from_numpy(arr).permute(2,0,1)[None].to(r.device).half()*2-1
            input_sha=hashlib.sha256(x.float().cpu().numpy().tobytes()).hexdigest();before=rh(rng())
            y=r.infer(x);torch.cuda.synchronize(r.device);cpu=y.detach().cpu().clone();state=rng()
            need(y.dtype==torch.float32 and y.device==x.device and y.shape==x.shape and bool(torch.isfinite(cpu).all()) and float(cpu.min())>=-1 and float(cpu.max())<=1,'Output contract differs')
            need(all(torch.equal(live.detach().cpu(),old) for live,old in held),'Retained previous output overwritten')
            held.append((y,cpu));ref=refdir/f'{a.suite}_reference_{case}.pt'
            ownpng=save_png(d/f'{name}_case{case}.png',cpu)
            row=dict(case=case,input_path=str(src),input_source_sha256=sha(src),input_float_sha256=input_sha,shape=list(x.shape),finite_range_contract_PASS=True,float_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),png=ownpng,pre_four_RNG_SHA256=before,post_four_RNG_SHA256=rh(state))
            if a.phase=='reference':
                need(not ref.exists(),'Immutable reference PT exists');torch.save(dict(y=cpu,rng=state,input_sha256=input_sha,png_sha256=ownpng['sha256']),ref)
            else:
                source_heads[str(ref)]=sha(ref);saved=torch.load(ref,map_location='cpu',weights_only=False)
                need(saved['input_sha256']==input_sha,'Reference input differs')
                refpng=save_png(d/f'{a.suite}_reference_case{case}.png',saved['y']);need(refpng['sha256']==saved['png_sha256'],'Regenerated reference PNG differs')
                delta=(cpu-saved['y']).abs();row.update(reference_png=refpng,float_equal=torch.equal(cpu,saved['y']),max_abs=float(delta.max()),mean_abs=float(delta.mean()),png_equal=ownpng['sha256']==refpng['sha256'],post_four_rng_equal=rh(state)==rh(saved['rng']),reference_PT_sha256=source_heads[str(ref)])
            rows.append(row);print(json.dumps(row),flush=True)
    need(all(sha(path)==digest for path,digest in source_heads.items()),'Read-only reference bytes changed')
    report=dict(**plan,status='complete',scope='Fixed visual diagnostic only; not full quality/latency or permission to reuse A7750 quality',output_ownership_preserved=True,constructor_four_RNG_equal_reference=True if a.phase=='candidate' else None,reused_reference_files_SHA256=source_heads,original_protocol=r.protocol,rows=rows)
    if a.phase=='candidate':report['all_float_png_four_RNG_exact']=all(z['float_equal'] and z['png_equal'] and z['post_four_rng_equal'] for z in rows)
    with result.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'result':str(result),'sha256':sha(result),'all_float_png_four_RNG_exact':report.get('all_float_png_four_RNG_exact')}),flush=True)


if __name__=='__main__':main()
