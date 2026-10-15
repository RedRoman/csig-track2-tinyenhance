"""Root-scheduled native four-case PixG5i diagnostic; default CPU plan only.

Unchanged registered Runner.infer; original FP16 input and rounded PNG encoding.
This is NOT the formal 103-image sequence, NOT a score, and NOT a speed test.
No validation image or GT is read. Use a new isolated diagnostic output folder.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013')
CASES=[7,16,41,83]
ADAPTER='70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a'
REFERENCE='6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'
PIX='2f1cd3cd8311381e432138f489b01b982e08ebd5b09ecbc05ac87f7bfb577923'
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'
FORMAL='1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'


def need(v,m):
    if not v:raise RuntimeError(m)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def record(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def read(p):return json.loads(Path(p).read_text())


def atomic_json(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def main():
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--model-dir',type=Path,required=True)
    q.add_argument('--candidate-manifest-sha256',required=True)
    q.add_argument('--checkpoint-sha256',required=True)
    q.add_argument('--test-dir',type=Path,default=R/'source_payload/assets/input')
    q.add_argument('--out-dir',type=Path,required=True)
    q.add_argument('--execute',action='store_true');a=q.parse_args()
    md=a.model_dir.resolve();out=a.out_dir.resolve();td=a.test_dir.resolve()
    need(out.is_relative_to(R/'analysis/pixg5i_diag4') and not out.exists(),'New diagnostic-only output required')
    cmfile=md.parent/'candidate_manifest.json';cm=read(cmfile);bfile=md/'pixg5i_binding.json';b=read(bfile)
    need(sha(cmfile)==a.candidate_manifest_sha256,'Actual registered candidate manifest differs')
    need(cm['schema']=='pixg5i_registered_candidate_v1' and cm['model_type']=='supplied_pixG5i_pixel_UNet' and
         Path(cm['model_dir']).resolve()==md and cm['used_as_G4_reference'] is False and
         cm['initial_package_checkpoint_sha256']==INITIAL and cm['stage'] in ['original_baseline','distilled_checkpoint'],
         'Actual original PixG5i family registration required')
    need(cm['checkpoint_sha256']==b['checkpoint_sha256']==a.checkpoint_sha256==sha(md/'model.pth'),
         'Actual checkpoint bytes differ')
    need(cm['stage']==b['stage'] and cm['runner_sha256']==sha(md/'runner.py')==ADAPTER and
         sha(md/'reference_runner.py')==REFERENCE and sha(md/'pix_model.py')==PIX,'Actual inference implementation changed')
    for n,v in cm['files'].items():need(sha(md/n)==v['sha256'],'Actual registered model-directory file changed')
    need(sha(R/'code/render_cuda_submission.py')==FORMAL,'Formal renderer changed; diagnostic encoding reference differs')
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None;tasks=[]
    for c in CASES:
        f=td/f'case{c}.jpg'
        with Image.open(f) as im:size=list(im.size)
        tasks.append(dict(split='test',case=c,input=str(f),input_sha256=sha(f),size=size))
    plan=dict(schema='pixg5i_native4_diagnostic_v1',scope='DIAGNOSTIC_NATIVE4_ONLY_NOT_FORMAL_OR_SCORE',
              model_dir=str(md),candidate_manifest=record(cmfile),binding=record(bfile),checkpoint_sha256=a.checkpoint_sha256,
              tasks=tasks,source=record(__file__),original_formal_renderer_sha256=FORMAL,
              input_encoding='original float32 RGB/255 -> half()*2-1; full native image',
              output_encoding='original float().clamp(-1,1); ((*0.5+0.5)*255).round().astype(uint8); PNG compress1',
              cases=CASES,seed=231,RNG_scope='Seed231 before model initialization, then continuous diagnostic order7/16/41/83; not full-val3/test100 sequence',
              extra_warmup=False,quality_or_speed_claim=False,no_val_or_GT=True)
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',plan=plan,out_dir=str(out))));return
    import torch
    import numpy as np
    need(torch.cuda.is_available(),'Root must schedule this diagnostic on a CUDA GPU')
    torch.cuda.set_device(0)
    def seed():random.seed(231);np.random.seed(231);torch.manual_seed(231);torch.cuda.manual_seed_all(231)
    def rng():return dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),device_rng=torch.cuda.get_rng_state(0))
    def rng_digest(value):
        def normalized(x):
            if torch.is_tensor(x):
                y=x.detach().cpu().contiguous().numpy();return dict(dtype=str(y.dtype),shape=list(y.shape),sha256=hashlib.sha256(y.tobytes()).hexdigest())
            if isinstance(x,np.ndarray):return dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.tobytes()).hexdigest())
            if isinstance(x,dict):return {k:normalized(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [normalized(v) for v in x]
            if isinstance(x,np.generic):return x.item()
            return x
        return hashlib.sha256(json.dumps(normalized(value),sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True);seed();before=rng();old=os.getcwd();sys.path.append(str(md))
    try:
        os.chdir(md);s=importlib.util.spec_from_file_location('_pixg5i_diag_submission_runner',md/'runner.py')
        module=importlib.util.module_from_spec(s);sys.modules[s.name]=module;s.loader.exec_module(module)
        runner=module.Runner(str(md));torch.cuda.synchronize()
    finally:os.chdir(old)
    need(runner.protocol['checkpoint_sha256']==a.checkpoint_sha256 and runner.protocol['runner_sha256']==ADAPTER,'Runtime identity differs')
    protocol=dict(plan,runner=runner.protocol,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                  model_dir_sha256={str(f.relative_to(md)):sha(f) for f in sorted(md.rglob('*')) if f.is_file()},
                  constructor_before_fourRNG_sha256=rng_digest(before),constructor_after_fourRNG_sha256=rng_digest(rng()))
    pid=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    atomic_json(out/'diagnostic_protocol.json',dict(protocol_id=pid,protocol=protocol));rows=[]
    for index,t in enumerate(tasks):
        src=Path(t['input']);need(sha(src)==t['input_sha256'],'Input changed')
        with Image.open(src) as im:array=np.asarray(im.convert('RGB'),dtype=np.float32)/255.0
        x=torch.from_numpy(array).permute(2,0,1)[None].to('cuda').half()*2-1
        pre=rng();began=time.time()
        with torch.no_grad():native=runner.infer(x)
        need(native.shape==x.shape and native.dtype==torch.float16 and native.device==x.device and
             bool(torch.isfinite(native).all()),'Invalid original native FP16 output')
        y=native.float().clamp(-1,1)
        image=Image.fromarray(((y[0].permute(1,2,0).cpu().numpy()*.5+.5)*255).round().astype(np.uint8))
        target=out/f'render/test/case{t["case"]}.png';target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.tmp');image.save(tmp,format='PNG',compress_level=1);os.replace(tmp,target)
        post=rng();row=dict(task_index=index,split='test',case=t['case'],input_sha256=t['input_sha256'],
                            output=str(target.relative_to(out)),PNG=record(target),size=t['size'],
                            before_fourRNG_sha256=rng_digest(pre),after_fourRNG_sha256=rng_digest(post),
                            seconds_diagnostic_only=time.time()-began,not_latency_measurement=True)
        rows.append(row)
        state=dict(protocol_id=pid,records=rows,**post);temp=out/'diagnostic_progress.tmp'
        with temp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        os.replace(temp,out/'diagnostic_progress.pt')
        atomic_json(out/'diagnostic_status.json',dict(status='rendering',completed=len(rows),total=4,latest=row))
        print(json.dumps(row),flush=True);del x,native,y,image,array;torch.cuda.empty_cache()
    for row in rows:need(sha(out/row['output'])==row['PNG']['sha256'],'Committed diagnostic PNG changed')
    need(sha(cmfile)==a.candidate_manifest_sha256 and sha(md/'model.pth')==a.checkpoint_sha256,'Source changed during diagnostic')
    atomic_json(out/'diagnostic_manifest.json',dict(status='COMPLETE_DIAGNOSTIC4_NOT_FULL_OR_SCORE',protocol_id=pid,
                protocol=protocol,records=rows,actual_native_count=4,quality_or_speed_claim=False))
    atomic_json(out/'diagnostic_status.json',dict(status='complete',completed=4,total=4,not_formal_or_score=True))
    (out/'diagnostic_complete').touch();print('DIAGNOSTIC4_COMPLETE',flush=True)


if __name__=='__main__':main()
