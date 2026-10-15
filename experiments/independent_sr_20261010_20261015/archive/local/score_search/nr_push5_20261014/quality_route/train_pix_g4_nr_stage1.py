"""New, bounded G4 pixel-student NR fine-tuning; never an HYPIR continuation.

Initial state is the user's supplied trained reference G4. Adam/RNG are new.
Frozen native A7750 PNG pseudo-labels, test-input crops only, no validation GT. Original
metric networks stay frozen. Exports normal cfg/state Pix checkpoints plus
immutable exact FP32/Adam/four-RNG resumes; no inference Runner is imported.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time

G4_SHA = '283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
PIX_SHA = '5b1ba75929d25c2663ec5aba5f7cd4d4a56b79f214dd8e7a49e48c34eca50f5f'
HELPER_SHA = '0998967aa6052ec9a5090219e17cb73c05468b06d7acc1eae27108c1c412fb4b'
TEACHER_MANIFEST_SHA = '35355c94cace390b5f5918cfb607d85b0f55c9cd1e1ba0a4f6a4d034abb54e16'
TEACHER_PROTOCOL_SHA = 'c0c2389b844b83ccda971f9179b04221f0a3751add428deef3bfc96ba80a0dc7'
TEACHER_RAW_SHA = '589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
METRIC_REFERENCE_SHA = '86015a361f5fa1429f1d797f175d27450cb6f798e1df4483badf01d876b2ab5b'
EXPECTED_CFG = dict(chs=(32,64,128), nbs=(1,2,4), norm='none', tdim=128, ps=2,
                    t_fixed=999.0, arch_ver=2)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def signature(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args()
    config = json.loads(known.config.read_text()) if known.config else {}
    p = argparse.ArgumentParser(description=__doc__, parents=[pre])
    p.add_argument('--root', type=Path, default=Path('/cache/independent_sr_cuda_20261013'))
    p.add_argument('--run')
    p.add_argument('--stage-steps', type=int, default=250)
    p.add_argument('--save-every', type=int, default=125)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--reward-weight', type=float, default=.01)
    p.add_argument('--musiq-weight', type=float, default=1.)
    p.add_argument('--clip-weight', type=float, default=.5)
    p.add_argument('--gradient-anchor', type=float, default=.3)
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--context-patch', type=int, default=1024)
    p.add_argument('--loss-patch', type=int, default=512)
    p.add_argument('--maniqa-samples', type=int, default=20)
    p.add_argument('--resume', type=Path)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--validate-config-only', action='store_true')
    allowed = {a.dest for a in p._actions}
    if set(config)-allowed:
        p.error('Unknown config fields: '+str(sorted(set(config)-allowed)))
    p.set_defaults(**config)
    a = p.parse_args()
    if not a.run or not re.fullmatch(r'[A-Za-z0-9_-]+', a.run) or not a.run.startswith('pix_push5_'):
        p.error('A new safe pix_push5_ run name is required')
    if not isinstance(a.stage_steps, int) or not 1 <= a.stage_steps <= 250 or a.save_every != 125:
        p.error('This new stage allows1..250 updates; saves every125 and final')
    if (a.context_patch, a.loss_patch, a.maniqa_samples) != (1024,512,20):
        p.error('First experiment fixes1024 generator context, central512 loss, MANIQA20')
    for name, lo, hi in [('lr',1e-7,2e-5),('reward_weight',.001,.05),
                         ('musiq_weight',.5,2.),('clip_weight',0.,1.),('gradient_anchor',.1,.6)]:
        x = getattr(a,name)
        if not isinstance(x,(int,float)) or not math.isfinite(x) or not lo <= x <= hi:
            p.error(f'{name} must be finite in [{lo},{hi}]')
    if not isinstance(a.seed,int) or not 0 <= a.seed < 2**32:
        p.error('Explicit integer32bit seed required')
    return a


def main():
    a = parse()
    if a.validate_config_only:
        print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},sort_keys=True))
        return
    root = Path(a.root).resolve()
    out = root/'adaptation/runs'/a.run
    reference_dir = root/'calibration/g4/model_dir'
    init = reference_dir/'model.pth'
    pix_source = reference_dir/'pix_model.py'
    helper_source = root/'code/cuda_training/train_cuda_nr.py'
    metric_reference = root/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/config.json'
    need(sha(init)==G4_SHA and sha(pix_source)==PIX_SHA, 'Supplied trained G4/reference source changed')
    need(sha(helper_source)==HELPER_SHA and sha(metric_reference)==METRIC_REFERENCE_SHA, 'Frozen metric helper/reference changed')
    ref = json.loads(metric_reference.read_text())['protocol']
    need(ref['metric_weights_sha256'] and len(ref['inputs'])==100, 'Frozen metric/input catalog missing')
    inputs = ref['inputs']
    need([x['case'] for x in inputs]==list(range(1,101)), 'Exactly test inputs1..100 required')
    need(all(Path(x['path']).resolve().parent==root/'source_payload/assets/input' for x in inputs), 'Training may only read test100 input JPEGs')
    sources = {str(p):sha(p) for p in [Path(__file__).resolve(),pix_source,helper_source,metric_reference,
        root/'code/numpy_legacy_compat.py',root/'code/maniqa_offline_init.py',root/'scoring_tool/csig_metrics.py']}
    for path in list(sources)[4:]:
        need(sources[path]==ref['source_sha256'][path], 'Frozen metric source changed: '+path)
    for item in inputs:
        need(sha(item['path'])==item['sha256'], 'Test input changed')
    weights = root/'scoring_tool/weights'
    for rel, expected in ref['metric_weights_sha256'].items():
        need(sha(weights/rel)==expected, 'Metric weight changed: '+rel)
    teacher_dir=root/'nr_push4_full_evaluations/push4_a7750_compiled_all'
    teacher_manifest=teacher_dir/'render_manifest.json'
    teacher_protocol=teacher_dir/'render_protocol.json'
    need(sha(teacher_manifest)==TEACHER_MANIFEST_SHA and sha(teacher_protocol)==TEACHER_PROTOCOL_SHA, 'Pinned native A7750 teacher catalog changed')
    manifest=json.loads(teacher_manifest.read_text());rp=json.loads(teacher_protocol.read_text())
    need(manifest['scope']=='test100+validation3' and manifest['protocol']==rp['protocol'], 'Teacher must be completed native full103')
    need(manifest['protocol_id']==rp['protocol_id']==hashlib.sha256(json.dumps(rp['protocol'],sort_keys=True).encode()).hexdigest(), 'Teacher protocol identity changed')
    need(rp['protocol']['runner']['checkpoint_sha256']==TEACHER_RAW_SHA, 'Teacher raw checkpoint identity changed')
    teacher_records=[r for r in manifest['records'] if r['split']=='test']
    need([r['case'] for r in teacher_records]==list(range(1,101)), 'Exactly teacher test100 required; validation excluded')
    tasks={t['case']:t for t in rp['protocol']['tasks'] if t['split']=='test'}
    from PIL import Image
    for item,record in zip(inputs,teacher_records):
        path=(teacher_dir/record['output']).resolve()
        need(path.parent==teacher_dir/'render/test' and sha(path)==record['sha256'], 'Actual offline teacher PNG SHA/path changed')
        need(tasks[item['case']]['input_sha256']==item['sha256'], 'Teacher/student source input differs')
        with Image.open(item['path']) as src,Image.open(path) as target:
            need(src.size==target.size==tuple(record['size']) and target.mode=='RGB' and target.format=='PNG', 'Teacher native dimensions/encoding differ')
        record['path']=str(path)
    teacher_signatures={r['path']:signature(r['path']) for r in teacher_records}
    guards = {str(p):signature(p) for p in [init,pix_source,helper_source,metric_reference,teacher_manifest,teacher_protocol]}
    import torch
    import numpy as np
    from PIL import Image
    need(torch.cuda.is_available() and 'torch_npu' not in sys.modules, 'Native CUDA required')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    Image.MAX_IMAGE_PIXELS = None
    helpers = load_source('_pix_g4_frozen_metric_helpers',helper_source)
    pix = load_source('_pix_g4_frozen_architecture',pix_source)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not (out/'config.json').exists() or a.resume is not None, 'Existing run requires explicit resume; never overwrite a prior run')
        helpers.seed_all(a.seed,torch,np)
        original = torch.load(init,map_location='cpu',weights_only=False)
        need(original['cfg']==EXPECTED_CFG and original['step']==30000, 'Sealed G4 architecture/reference step changed')
        need(len(original['state'])==100 and sum(v.numel() for v in original['state'].values())==3163404, 'G4 parameter count changed')
        need(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in original['state'].values()), 'Reference must hold finite exact FP32 states')
        student = pix.build_pix(init,device=device,dtype=torch.float32).eval().requires_grad_(True)
        named = list(student.named_parameters())
        need(len(named)==100 and [k for k,_ in named]==list(original['state']), 'Trainable/state order differs')
        for model in [student]:
            current=model.state_dict()
            need(all(v.dtype==torch.float32 and torch.equal(v.detach().cpu(),original['state'][k]) for k,v in current.items()), 'Initial FP32 state not exact')
        params = [p for _,p in named]
        metrics, hidden_clip, maniqa_proof, metric_sha = helpers.create_metrics(root,device,a.maniqa_samples)
        need(metric_sha==ref['metric_weights_sha256'], 'Actual frozen metric identities differ')
        protocol = dict(schema='pix_g4_A7750_offline_distill_nr_stage1_v1',initial_reference_checkpoint=str(init),
            initial_reference_sha256=G4_SHA,reference_training_step=30000,
            origin='New training initialized from the user-supplied trained G4 reference; not our original reference training and not HYPIR continuation',
            reference_original_args=original.get('args'),architecture_cfg=original['cfg'],state_tensors=100,parameters=3163404,
            optimizer='new AdamW FP32, weight_decay0, foreachFalse; initial optimizer step0; no original reference optimizer supplied',
            seed=a.seed,seed_scope='Reset all4 RNG after student/teacher/metric construction; continuous training RNG thereafter',
            lr=a.lr,reward_weight=a.reward_weight,musiq_weight=a.musiq_weight,clip_weight=a.clip_weight,
            gradient_anchor=a.gradient_anchor,fidelity_L1_weight=1.,fidelity_LPIPS_weight=.1,
            loss='L1(A7750_native_PNG)+.1*LPIPS(A7750_native_PNG)+gradient_anchor*spatial_gradient_L1 - reward_weight*(musiq_weight*MUSIQ/100+MANIQA+clip_weight*CLIPIQA)',
            teacher='Frozen original native A7750 test100 PNGs, aligned same case/coordinates; no online self-anchor',teacher_raw_sha256=TEACHER_RAW_SHA,
            teacher_manifest=str(teacher_manifest),teacher_manifest_sha256=TEACHER_MANIFEST_SHA,
            teacher_protocol_sha256=TEACHER_PROTOCOL_SHA,teacher_records=teacher_records,
            model_precision='unfolded FP32 student; offline teacher PNG FP32 RGB/255; no autocast/TF32; formal FP16 folded deployment must be independently rescored',
            input_alignment='tool FP16[-1,1] roundtrip then FP32',output_alignment='core[-1,1] -> RGB clamp; no wavelet or postprocessing',
            context_patch=a.context_patch,loss_patch=a.loss_patch,context_margin=256,crop_grid=8,
            context_scope='1024 generator context, central512 metrics/fidelity; positions aligned to native unshuffle/downsample grid; no GT',
            maniqa_train_samples=20,maniqa_formal_samples=100,niqe='formal evaluation only; no train gradient',
            training_data='100 fixed test JPEG inputs only; no validation LQ/GT used for training',inputs=inputs,
            metric_weights_sha256=metric_sha,source_sha256=sources,init_reference_config_for_metric_hashes=str(metric_reference),
            trainable_parameter_names=[k for k,_ in named],
            flags=dict(cudnn_benchmark=False,cudnn_allow_tf32=False,matmul_allow_tf32=False),
            versions={n:__import__('importlib.metadata',fromlist=['version']).version(n) for n in ['torch','numpy','Pillow','pyiqa']})
        identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.,foreach=False)
        helpers.seed_all(a.seed,torch,np)
        done=0
        if a.resume:
            resume=Path(a.resume).resolve()
            need(resume.parent==out.resolve(), 'Resume must belong to this new run')
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            need(saved['protocol_id']==identity and saved['optimizer_step']==saved['local_step'], 'Resume protocol/step differs')
            need(list(saved['state'])==list(original['state']) and all(v.dtype==torch.float32 for v in saved['state'].values()), 'Resume FP32 state schema differs')
            student.load_state_dict(saved['state'],strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            need((len(optimizer.state)==0 if saved['local_step']==0 else len(optimizer.state)==100 and {int(v['step']) for v in optimizer.state.values()}=={saved['local_step']}), 'Resume Adam counters differ')
            done=saved['local_step']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],0)
            del saved
        need(0<=done<=a.stage_steps, 'Resume outside this stage')
        helpers.atomic_json(out/'config.json',dict(protocol_id=identity,protocol=protocol,target_stage_steps=a.stage_steps,
            target_optimizer_step=a.stage_steps,new_training_steps=a.stage_steps,save_every=a.save_every,
            initial_optimizer_step=0,original_reference_training_step=30000))
        helpers.atomic_json(out/'maniqa_initial_state_proof.json',maniqa_proof)
        helpers.atomic_json(out/'initial_fp32_proof.json',dict(status='PASS',offline_A7750_teacher_100_actual_PNG_SHA_verified=True,
            teacher_raw_sha256=TEACHER_RAW_SHA,teacher_manifest_sha256=TEACHER_MANIFEST_SHA,
            initial_student_exact_original_FP32=True,reference_sha256=G4_SHA,new_Adam_initial_step=0,
            reference_optimizer_not_available_and_not_inherited=True,all100_FP32_tensors=True,
            no_validation_GT_training=True,architecture_changed_from_HYPIR=True))
        initial = {k:v.clone() for k,v in original['state'].items()}
        del original
        if done==0 and not a.preflight_only and a.resume is None:
            raw0=out/'step0.pth'
            need(not raw0.exists() and not (out/'resume_step0.pt').exists(), 'Never overwrite initial saved state')
            helpers.atomic_torch(raw0,dict(cfg=dict(student.cfg),state=initial,step=0,reference_step=30000,
                origin='Initial supplied G4, zero new updates',protocol_id=identity),torch)
            saved0=dict(protocol_id=identity,cfg=dict(student.cfg),state=initial,optimizer=optimizer.state_dict(),
                local_step=0,optimizer_step=0,initial_reference_sha256=G4_SHA,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(0))
            helpers.atomic_torch(out/'resume_latest.pt',saved0,torch)
            os.link(out/'resume_latest.pt',out/'resume_step0.pt')
            m=out/'resume_step0.pt'
            helpers.atomic_json(out/'checkpoint_step0.json',dict(local_step=0,optimizer_step=0,
                raw_checkpoint=str(raw0),raw_sha256=sha(raw0),resume_checkpoint=str(m),resume_sha256=sha(m),
                resume_bytes=m.stat().st_size,resume_mtime_ns=m.stat().st_mtime_ns,resume_inode=m.stat().st_ino,
                changed_FP32_state_tensors=0,all100_FP32_finite=True,all4_RNG_saved=True,new_Adam_state_entries=0))
            del saved0
        begin=time.time();first=True
        with (out/'train.jsonl').open('a',buffering=1) as log:
            for local_step in range(done+1,a.stage_steps+1):
                started=time.time();case=random.randint(1,100);item=inputs[case-1]
                need(sha(item['path'])==item['sha256'], 'Training input changed')
                with Image.open(item['path']) as image:
                    image=image.convert('RGB');w,h=image.size
                    need(min(w,h)>=a.context_patch, 'Input smaller than required context')
                    left=8*random.randrange((w-a.context_patch)//8+1)
                    top=8*random.randrange((h-a.context_patch)//8+1)
                    array=np.asarray(image.crop((left,top,left+a.context_patch,top+a.context_patch)),dtype=np.float32).copy()/255.
                x=torch.from_numpy(array).permute(2,0,1)[None].to(device)
                xm11=(x.half()*2-1).float()
                optimizer.zero_grad(set_to_none=True)
                pred=((student.core(xm11).float()+1)*.5).clamp(0,1)
                teacher_record=teacher_records[case-1]
                need(sha(teacher_record['path'])==teacher_record['sha256'], 'Offline teacher PNG changed')
                with Image.open(teacher_record['path']) as image:
                    target_array=np.asarray(image.crop((left+256,top+256,left+768,top+768)),dtype=np.float32).copy()/255.
                target=torch.from_numpy(target_array).permute(2,0,1)[None].to(device)
                rgb=pred[:,:,256:768,256:768]
                lpips=metrics['lpips'](rgb,target).mean()
                l1=(rgb-target).abs().mean();spatial=helpers.gradient_loss(rgb,target)
                fidelity=l1+.1*lpips+a.gradient_anchor*spatial
                scores={n:metrics[n](rgb).mean()/(100 if n=='musiq' else 1) for n in ['musiq','maniqa','clipiqa']}
                reward=a.musiq_weight*scores['musiq']+scores['maniqa']+a.clip_weight*scores['clipiqa']
                loss=fidelity-a.reward_weight*reward
                need(bool(torch.isfinite(loss)), 'Nonfinite loss')
                component_grads={}
                if first:
                    for name,value in scores.items():
                        grad=torch.autograd.grad(value,rgb,retain_graph=True)[0]
                        norm=float(grad.float().norm())
                        need(bool(torch.isfinite(grad).all()) and norm>0, 'True frozen NR input gradient missing: '+name)
                        component_grads[name]=dict(input_gradient_norm=norm,finite=True,nonzero=True)
                loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(params,1.)
                finite=sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
                nonzero=sum(p.grad is not None and bool(p.grad.abs().max()>0) for p in params)
                need(bool(torch.isfinite(gradient_norm)) and finite==100 and nonzero==100, 'All100 real student gradients must be finite and nonzero')
                need(not target.requires_grad and target.grad_fn is None, 'Offline teacher must remain detached')
                need(not any(p.grad is not None for m in list(metrics.values())+[hidden_clip] for p in m.parameters()), 'Frozen metric got gradient')
                torch.cuda.synchronize()
                row=dict(local_step=local_step,optimizer_step=local_step,case=case,context_crop_xy=[left,top],
                    loss_crop_xy=[left+256,top+256],context_patch=1024,loss_patch=512,
                    teacher_png=teacher_record['path'],teacher_png_sha256=teacher_record['sha256'],
                    loss=float(loss.detach()),fidelity=float(fidelity.detach()),L1=float(l1.detach()),LPIPS=float(lpips.detach()),
                    spatial_gradient_L1=float(spatial.detach()),reward=float(reward.detach()),
                    scores={k:float(v.detach()) for k,v in scores.items()},gradient_norm=float(gradient_norm),
                    finite_parameter_gradients=finite,nonzero_parameter_gradients=nonzero,lr=optimizer.param_groups[0]['lr'],
                    step_seconds=time.time()-started,elapsed_seconds=time.time()-begin,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                if first:
                    helpers.atomic_json(out/'gradient_preflight.json',dict(status='PASS',components=component_grads,row=row,
                        no_optimizer_update_yet=True,frozen_metrics_and_offline_teacher=True,expected_initial_fidelity_zero=False))
                    first=False
                if a.preflight_only:
                    helpers.atomic_json(out/'status.json',dict(stage='preflight_complete',optimizer_updates=0,row=row))
                    print(json.dumps(row),flush=True);return
                optimizer.step();log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                helpers.atomic_json(out/'status.json',dict(stage='training',completed_local_steps=local_step,total=a.stage_steps,row=row))
                if local_step%a.save_every==0 or local_step==a.stage_steps:
                    need(len(optimizer.state)==100 and {int(v['step']) for v in optimizer.state.values()}=={local_step}, 'Saved Adam step/schema differs')
                    need(all(v[k].dtype==torch.float32 and bool(torch.isfinite(v[k]).all()) for v in optimizer.state.values() for k in ['exp_avg','exp_avg_sq']), 'Saved Adam moments must be finite FP32')
                    state={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
                    need(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()), 'Saved model nonfinite/nonFP32')
                    changed=sum(not torch.equal(v,initial[k]) for k,v in state.items())
                    need(changed==100, 'Actual training must change all100 state tensors')
                    raw=out/f'step{local_step}.pth'
                    raw_data=dict(cfg=dict(student.cfg),state=state,step=local_step,reference_step=30000,
                        origin='Fine-tuned from supplied trained G4 reference using offline A7750 PNG pseudo-labels; new optimizer/training stage',protocol_id=identity)
                    helpers.atomic_torch(raw,raw_data,torch)
                    saved=dict(protocol_id=identity,cfg=dict(student.cfg),state=state,optimizer=optimizer.state_dict(),
                        local_step=local_step,optimizer_step=local_step,initial_reference_sha256=G4_SHA,
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(0))
                    latest=out/'resume_latest.pt';helpers.atomic_torch(latest,saved,torch)
                    milestone=out/f'resume_step{local_step}.pt';need(not milestone.exists(), 'Never overwrite immutable saved resume')
                    os.link(latest,milestone)
                    marker=dict(local_step=local_step,optimizer_step=local_step,raw_checkpoint=str(raw),raw_sha256=sha(raw),
                        resume_checkpoint=str(milestone),resume_sha256=sha(milestone),resume_bytes=milestone.stat().st_size,
                        resume_mtime_ns=milestone.stat().st_mtime_ns,resume_inode=milestone.stat().st_ino,
                        changed_FP32_state_tensors=changed,all100_FP32_finite=True,all4_RNG_saved=True)
                    helpers.atomic_json(out/f'checkpoint_step{local_step}.json',marker)
                    del state,saved,raw_data
                del x,xm11,pred,rgb,target,lpips,l1,spatial,fidelity,scores,reward,loss
        for path,before in guards.items():need(signature(path)==before, 'Sealed G4/reference file changed during training')
        for path,before in teacher_signatures.items():need(signature(path)==before, 'Offline teacher PNG changed during training')
        for path,expected in sources.items():need(sha(path)==expected, 'Source changed during training')
        helpers.atomic_json(out/'status.json',dict(stage='complete',completed_local_steps=a.stage_steps,
            optimizer_step=a.stage_steps,requires_independent_full103_418_and_same_card_latency=True,
            original_reference_sha256=G4_SHA,origin='Supplied trained G4 initialized; new real fine-tuning'))
        (out/f'training_complete_step{a.stage_steps}').touch()


if __name__=='__main__':
    main()
