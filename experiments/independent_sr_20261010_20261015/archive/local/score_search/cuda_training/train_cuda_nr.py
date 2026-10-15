"""Isolated raw-HYPIR CUDA NR warm start; no fused/compiled inference Runner.

Frozen current6000/coeff175 anchor, identical latent; real frozen FP32
MUSIQ/MANIQA/CLIPIQA input gradients, LPIPS/L1/gradient fidelity. New AdamW
and seed are explicit. This file never imports torch_npu or changes old runs.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import time

INIT_SHA = '2c82a71a94aa754bd4a26a2a474c2bbe4989bde1d38d2e15d9e58eb0c3e408bc'
MODULES = 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj'
NAMES = ['musiq', 'maniqa', 'clipiqa', 'lpips']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):h.update(b)
    return h.hexdigest()


def atomic_json(path, data):
    p=path.with_suffix('.tmp')
    with p.open('w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    p.replace(path)


def atomic_torch(path, data, torch):
    p=path.with_suffix('.tmp')
    with p.open('wb') as f:
        torch.save(data,f);f.flush();os.fsync(f.fileno())
    p.replace(path)


def seed_all(seed, torch, np):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def gradient_loss(x,y):
    return .5*((x[:,:,:,1:]-x[:,:,:,:-1]-y[:,:,:,1:]+y[:,:,:,:-1]).abs().mean()+
               (x[:,:,1:,:]-x[:,:,:-1,:]-y[:,:,1:,:]+y[:,:,:-1,:]).abs().mean())


def create_metrics(root, device, maniqa_samples):
    import torch
    sys.path.insert(0,str(root/'code'));sys.path.insert(0,str(root/'scoring_tool'))
    import numpy_legacy_compat  # dtype catalog only; no metric or RNG modification
    import maniqa_offline_init
    import csig_metrics as CM
    weights=root/'scoring_tool/weights'
    os.environ['TORCH_HOME']=str(weights/'torch');torch.hub.set_dir(str(weights/'torch/hub'))
    CM.ensure_weights()
    maniqa_offline_init.install(weights/'torch/hub/checkpoints/ckpt_koniq10k.pt')
    import pyiqa
    metrics={name:pyiqa.create_metric(CM.METRIC_NAMES[name],device=device,
                                    as_loss=True,loss_reduction='none') for name in NAMES}
    for m in metrics.values():m.eval().requires_grad_(False)
    metrics['maniqa'].net.test_sample=maniqa_samples
    clip=metrics['clipiqa'].net.clip_model[0]
    clip.float().to(device).eval().requires_grad_(False)  # hidden Python-list module
    proof=maniqa_offline_init.verify_final_state(metrics['maniqa'].net)
    if not proof['all_final_parameters_and_buffers_equal_checkpoint']:
        raise RuntimeError('MANIQA final state differs from supplied original checkpoint')
    for m in list(metrics.values())+[clip]:
        if any(p.requires_grad for p in m.parameters()) or any(v.training for v in m.modules()):
            raise RuntimeError('Metric must remain frozen/eval, including hidden CLIP')
    files=['torch/hub/checkpoints/musiq_koniq_ckpt-e95806b9.pth',
           'torch/hub/checkpoints/ckpt_koniq10k.pt','clip/RN50.pt',
           'torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth',
           'torch/hub/checkpoints/alexnet-owt-7be5be79.pth']
    weight_sha={f:sha(weights/f) for f in files}
    return metrics,clip,proof,weight_sha


def parse():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--config',type=Path)
    known,_=pre.parse_known_args();cfg=json.loads(known.config.read_text()) if known.config else {}
    p=argparse.ArgumentParser(description=__doc__,parents=[pre])
    p.add_argument('--root',type=Path,default=Path('/cache/independent_sr_cuda_20261013'))
    p.add_argument('--run');p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--anchor-checkpoint',type=Path);p.add_argument('--expected-init-sha',default=INIT_SHA)
    p.add_argument('--base-step',type=int,default=6000);p.add_argument('--coeff-t',type=int,default=175)
    p.add_argument('--segment-steps',type=int,default=100);p.add_argument('--seed',type=int,default=20261013)
    p.add_argument('--lr',type=float,default=2e-7);p.add_argument('--reward-weight',type=float,default=.075)
    p.add_argument('--clip-weight',type=float,default=.1);p.add_argument('--gradient-anchor',type=float,default=.3)
    p.add_argument('--maniqa-samples',type=int,default=20);p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--nr-round-ste',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--resume',type=Path);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--preflight-case',type=int)
    allowed={a.dest for a in p._actions}
    if set(cfg)-allowed:p.error('Unknown config fields: '+str(sorted(set(cfg)-allowed)))
    p.set_defaults(**cfg);a=p.parse_args()
    if not a.run or not re.fullmatch(r'[A-Za-z0-9_-]+',a.run):p.error('A new safe --run name is required')
    if not 1<=a.segment_steps<=250 or a.save_every<1:p.error('Bounded segment must be1..250, save_every positive')
    if a.preflight_case and not 1<=a.preflight_case<=100:p.error('preflight case must be1..100')
    return a


def main():
    a=parse();root=Path(a.root).resolve();out=root/'adaptation/runs'/a.run
    repo=root/'repos/HYPIR_cuda';base=Path('/temp/zihaohe_stage/baselines/sd21b')
    init=Path(a.init_checkpoint or root/'source_payload/adaptation/runs/ascend_balanced/step6000.pth').resolve()
    anchor_path=Path(a.anchor_checkpoint or init).resolve()
    if sha(init)!=a.expected_init_sha:raise RuntimeError('Initial raw checkpoint SHA mismatch')
    sys.path.insert(0,str(repo))
    import torch
    import numpy as np
    from PIL import Image
    from HYPIR.enhancer.sd2 import SD2Enhancer
    from HYPIR.utils.common import wavelet_reconstruction
    from torch.nn import functional as F
    if not torch.cuda.is_available() or 'torch_npu' in sys.modules:raise RuntimeError('Native CUDA only')
    torch.cuda.set_device(0);device=torch.device('cuda:0');torch.set_num_threads(4)
    Image.MAX_IMAGE_PIXELS=None
    input_records=[]
    for i in range(1,101):
        p=root/'source_payload/assets/input'/f'case{i}.jpg'
        with Image.open(p) as image:
            if min(image.size)<512:raise RuntimeError('Training input smaller than512')
        input_records.append(dict(case=i,path=str(p),sha256=sha(p)))
    # Recheck prior provenance before fresh initialization, including all base bytes.
    provenance=json.loads((root/'calibration/base_model_provenance_shared.json').read_text())
    for rel,record in provenance['initial_core_asset_hashes'].items():
        if sha(base/rel)!=record['sha256']:raise RuntimeError('Base core asset changed: '+rel)
    for rel,record in provenance['all_config_tokenizer_scheduler_metadata_hashes'].items():
        if sha(base/rel)!=record['sha256']:raise RuntimeError('Base metadata changed: '+rel)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_file=out/'resume_latest.pt'
        if state_file.exists() and not a.resume:raise RuntimeError('Existing training state requires explicit --resume')
        seed_all(a.seed,torch,np)
        def enhancer(weight):
            m=SD2Enhancer(base_model_path=str(base),weight_path=str(weight),lora_modules=MODULES.split(','),
                          lora_rank=256,model_t=200,coeff_t=a.coeff_t,device=str(device))
            m.init_models();m.G.eval().requires_grad_(False);m.vae.eval().requires_grad_(False)
            m.text_encoder.eval().requires_grad_(False);m.prepare_inputs(1,'')
            for name,p in m.G.named_parameters():
                if 'lora' in name:p.data=p.data.float()
            return m
        student=enhancer(init);teacher=enhancer(anchor_path)
        params=[]
        for name,p in student.G.named_parameters():
            if 'lora' in name:p.requires_grad_(True);params.append(p)
        if not params:raise RuntimeError('No LoRA trainable parameters')
        student.G.enable_gradient_checkpointing()
        metrics,clip,maniqa_proof,metric_sha=create_metrics(root,device,a.maniqa_samples)
        sources=[Path(__file__),repo/'HYPIR/enhancer/sd2.py',repo/'HYPIR/enhancer/base.py',repo/'HYPIR/utils/common.py',
                 root/'code/numpy_legacy_compat.py',root/'code/maniqa_offline_init.py',root/'scoring_tool/csig_metrics.py']
        protocol=dict(schema='cuda_raw_hypir_nr_training_v1',init_checkpoint=str(init),init_sha256=sha(init),
                      anchor_checkpoint=str(anchor_path),anchor_sha256=sha(anchor_path),base_step=a.base_step,
                      optimizer='new AdamW weight_decay0 foreachFalse; FP32 LoRA params',lr=a.lr,seed=a.seed,
                      seed_scope='seed before both model initialization; reset same seed AFTER all metrics initialization; then continuous crop/latent/metric RNG',
                      model_t=200,coeff_t=a.coeff_t,rank=256,batch=1,patch=512,latent='posterior.sample once, same detached z for frozen anchor',
                      model_precision='BF16 base; FP32 trainable LoRA; BF16 autocast generator/decoder',
                      metric_precision='FP32, autocast disabled; original networks eval/as_lossTrue/loss_reductionNone',
                      input_alignment='tool FP16[-1,1] roundtrip on GPU, scale1 bicubic before VAE',
                      output_alignment='decoded RGB -> native bicubic antialias -> original differentiable wavelet_reconstruction(decoded,lq) -> clamp',
                      nr_round_ste=a.nr_round_ste,quantization='round-to-uint8/255 forward with identity STE gradient' if a.nr_round_ste else 'unencoded RGB; no quantization surrogate',
                      reward_weight=a.reward_weight,clip_weight=a.clip_weight,gradient_anchor=a.gradient_anchor,
                      loss='L1(anchor)+.1*LPIPS(anchor)+gradient_anchor*spatial_gradient_L1 - reward_weight*(MUSIQ/100+MANIQA+clip_weight*CLIPIQA)',
                      maniqa_train_samples=a.maniqa_samples,maniqa_formal_score_samples=100,
                      niqe='evaluation only; not in training gradient',training_data='only test100 random512 crops; validationGT excluded',
                      anchor='frozen raw current6000, unfused/uncompiled, same latent and coefficient',
                      compile=False,fuse=False,inputs=input_records,metric_weights_sha256=metric_sha,
                      source_sha256={str(p):sha(p) for p in sources},base_provenance=provenance,
                      versions={name:__import__('importlib.metadata',fromlist=['version']).version(name) for name in
                                ['torch','torchvision','diffusers','transformers','peft','pyiqa','numpy']})
        identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0,foreach=False)
        # pyiqa constructors reset seeds to123; never let this silently select training crops.
        seed_all(a.seed,torch,np);done=0
        if a.resume:
            saved=torch.load(Path(a.resume),map_location='cpu',weights_only=False)
            if saved['protocol_id']!=identity:raise RuntimeError('Resume training protocol changed')
            current=student.G.state_dict()
            if set(saved['lora'])!={k for k in current if 'lora' in k}:raise RuntimeError('Resume LoRA keys changed')
            with torch.no_grad():
                for k,v in saved['lora'].items():current[k].copy_(v.to(current[k]))
            opt.load_state_dict(saved['optimizer']);done=saved['local_step']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'],0)
            del saved
        atomic_json(out/'config.json',dict(protocol_id=identity,protocol=protocol,target_segment_steps=a.segment_steps,
                                         target_global_step=a.base_step+a.segment_steps,save_every=a.save_every))
        atomic_json(out/'maniqa_initial_state_proof.json',maniqa_proof)
        begin=time.time();first=True
        with (out/'train.jsonl').open('a',buffering=1) as log:
            for local_step in range(done+1,a.segment_steps+1):
                start=time.time();case=a.preflight_case if a.preflight_only and a.preflight_case else random.randint(1,100)
                source=Path(input_records[case-1]['path'])
                if sha(source)!=input_records[case-1]['sha256']:raise RuntimeError('Training input changed')
                with Image.open(source) as image:
                    image=image.convert('RGB');w,h=image.size;left=random.randrange(w-511);top=random.randrange(h-511)
                    array=np.asarray(image.crop((left,top,left+512,top+512)),dtype=np.float32).copy()/255
                x=torch.from_numpy(array).permute(2,0,1)[None].to(device)
                x=((x.half()*2-1).float()+1)*.5
                x=F.interpolate(x,scale_factor=1,mode='bicubic')
                with torch.no_grad():z=student.vae.encode((x*2-1).to(torch.bfloat16)).latent_dist.sample()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=(student.vae.decode(student.forward_generator(z).to(torch.bfloat16)).sample.float()+1)/2
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    target=(teacher.vae.decode(teacher.forward_generator(z.detach()).to(torch.bfloat16)).sample.float()+1)/2
                pred=F.interpolate(pred,size=x.shape[-2:],mode='bicubic',antialias=True)
                pred=wavelet_reconstruction(pred,x)
                with torch.no_grad():
                    target=F.interpolate(target,size=x.shape[-2:],mode='bicubic',antialias=True)
                    target=wavelet_reconstruction(target,x)
                rgb=pred.clamp(0,1);target=target.clamp(0,1)
                metric_rgb=rgb+(torch.round(rgb*255)/255-rgb).detach() if a.nr_round_ste else rgb
                with torch.autocast('cuda',enabled=False):
                    lpips=metrics['lpips'](rgb,target).mean()
                    fidelity=(rgb-target).abs().mean()+.1*lpips+a.gradient_anchor*gradient_loss(rgb,target)
                    scores={n:metrics[n](metric_rgb).mean()/(100 if n=='musiq' else 1) for n in NAMES[:3]}
                    reward=scores['musiq']+scores['maniqa']+a.clip_weight*scores['clipiqa'];loss=fidelity-a.reward_weight*reward
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                audit=None
                if first:
                    audit={}
                    for name,value in scores.items():
                        g=torch.autograd.grad(value,rgb,retain_graph=True)[0]
                        norm=float(g.float().norm().item())
                        if not torch.isfinite(g).all() or norm<=0:raise RuntimeError('No finite nonzero real input gradient: '+name)
                        audit[name]=dict(value=float(value.detach()),input_gradient_norm=norm,input_gradient_finite=True)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
                nonzero=sum(p.grad is not None and bool(p.grad.abs().max()>0) for p in params)
                if not torch.isfinite(norm) or nonzero==0:raise RuntimeError('Nonfinite/zero LoRA gradient')
                if any(p.grad is not None for p in teacher.G.parameters()):raise RuntimeError('Frozen anchor got parameter gradient')
                if any(p.grad is not None for m in list(metrics.values())+[clip] for p in m.parameters()):
                    raise RuntimeError('Frozen metric got parameter gradient')
                torch.cuda.synchronize();row=dict(local_step=local_step,global_step=a.base_step+local_step,case=case,crop_xy=[left,top],
                    loss=float(loss.detach()),fidelity=float(fidelity.detach()),reward=float(reward.detach()),
                    scores={k:float(v.detach()) for k,v in scores.items()},gradient_norm=float(norm),
                    nonzero_lora_gradients=nonzero,step_seconds=time.time()-start,elapsed_seconds=time.time()-begin,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                if first:
                    atomic_json(out/'gradient_preflight.json',dict(status='passed',components=audit,row=row,
                                no_optimizer_update_yet=True,frozen_metrics=True,frozen_same_latent_anchor=True))
                    first=False
                if a.preflight_only:
                    atomic_json(out/'status.json',dict(stage='preflight_complete',row=row,optimizer_updates=0));print(json.dumps(row),flush=True);return
                opt.step();log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                atomic_json(out/'status.json',dict(stage='training',completed_local_steps=local_step,total=a.segment_steps,row=row))
                if local_step%a.save_every==0 or local_step==a.segment_steps:
                    lora={k:v.detach().cpu().clone() for k,v in student.G.state_dict().items() if 'lora' in k}
                    atomic_torch(out/f'step{a.base_step+local_step}.pth',lora,torch)
                    state=dict(protocol_id=identity,lora=lora,optimizer=opt.state_dict(),local_step=local_step,
                               global_step=a.base_step+local_step,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                               torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0))
                    atomic_torch(state_file,state,torch);del state,lora
                del x,z,pred,rgb,metric_rgb,target,loss,reward,scores,lpips,fidelity
        atomic_json(out/'status.json',dict(stage='complete',completed_local_steps=a.segment_steps,
                                         global_step=a.base_step+a.segment_steps,requires_independent_quality_evaluation=True))
        (out/f'training_complete_step{a.base_step+a.segment_steps}').touch()


if __name__=='__main__':
    main()
