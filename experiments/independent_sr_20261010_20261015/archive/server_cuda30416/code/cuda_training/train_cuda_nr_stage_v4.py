"""Bounded CUDA NR continuation of sealed best6500, preserving FP32 Adam/RNG.

Imports only helpers from unchanged v1 source. Own parser and stage loop;
exact FP32 teacher6500 or teacher6250, local1..1000/global6500+local.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import time

PARENT_PROTOCOL = '995cae4104717ed2e578cb0867efae4171a36d97b281925534fd1f74f662bff1'
PARENT_RESUME_SHA = '1f8d6991b490ebe9438b7e7b62c8ad958bac98979516d405f97a829c3c46461f'
PARENT_CONFIG_SHA = 'c6beaad0a6427a3eaaf3e829c04a25982c7ddbcb2df5c157f0aa37df686e0c5a'
TEACHER6250_PROTOCOL = 'fa5ea19179561aac206b26f7dd3596406f1ad376dccd38f49821146623b98d52'
TEACHER6250_RESUME_SHA = '3dcf896028f41c7c4c6e15074cc69c4cdb2ff7679e6756f5ee57a9a657f61c1f'
TEACHER6250_CONFIG_SHA = '2a459e95e248d1b502b651753b9598ce7adfb92c24e48863458265458ea1ca5a'
V1_SOURCE_SHA = '0998967aa6052ec9a5090219e17cb73c05468b06d7acc1eae27108c1c412fb4b'
V2_SOURCE_SHA = '9ac86a8d78ba0fc085b60278a460ecaf1ffef74d0dd4e21c4b5c5d84063c4651'
V3_SOURCE_SHA = '539844657210938295aa55eb71a2620bad40b974a9ae48ad7238ae7d70fa7c2c'
INIT_SHA = '2c82a71a94aa754bd4a26a2a474c2bbe4989bde1d38d2e15d9e58eb0c3e408bc'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def parse():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--config',type=Path)
    known,_=pre.parse_known_args();cfg=json.loads(known.config.read_text()) if known.config else {}
    p=argparse.ArgumentParser(description=__doc__,parents=[pre])
    p.add_argument('--root',type=Path,default=Path('/cache/independent_sr_cuda_20261013'))
    p.add_argument('--run',required=False)
    p.add_argument('--parent-resume',type=Path);p.add_argument('--parent-config',type=Path)
    p.add_argument('--expected-parent-resume-sha',default=PARENT_RESUME_SHA)
    p.add_argument('--expected-parent-config-sha',default=PARENT_CONFIG_SHA)
    p.add_argument('--expected-parent-protocol',default=PARENT_PROTOCOL)
    p.add_argument('--stage-steps',type=int,default=750);p.add_argument('--save-every',type=int,default=250)
    p.add_argument('--nr-round-ste',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--teacher',choices=['parent6500','parent6250'],default='parent6500')
    p.add_argument('--lr',type=float,default=2e-7)
    p.add_argument('--reward-weight',type=float,default=.075)
    p.add_argument('--clip-weight',type=float,default=.1)
    p.add_argument('--musiq-weight',type=float,default=1.5)
    p.add_argument('--gradient-anchor',type=float,default=.3)
    p.add_argument('--resume',type=Path);p.add_argument('--preflight-only',action='store_true')
    allowed={x.dest for x in p._actions}
    if set(cfg)-allowed:p.error('Unknown config fields: '+str(sorted(set(cfg)-allowed)))
    p.set_defaults(**cfg);a=p.parse_args()
    if not a.run or not re.fullmatch(r'[A-Za-z0-9_-]+',a.run):p.error('Safe new --run required')
    if not a.parent_resume or not a.parent_config:p.error('Explicit parent resume/config required')
    if not isinstance(a.stage_steps,int) or not 1<=a.stage_steps<=1000:
        p.error('New stage must add1..1000 updates')
    if a.save_every!=250:p.error('V4 saves raw checkpoints and replaces resume_latest every250 updates, plus final')
    if a.nr_round_ste:p.error('V4 branches require nr_round_ste=False')
    if a.teacher not in ['parent6500','parent6250']:p.error('Unsupported sealed teacher')
    bounds={'lr':(5e-8,4e-7),'reward_weight':(.025,.3),'clip_weight':(0.,1.),
            'musiq_weight':(.5,2.5),'gradient_anchor':(.1,.6)}
    for name,(lo,hi) in bounds.items():
        value=getattr(a,name)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
            p.error(f'{name} must be finite in [{lo},{hi}]')
    if (a.expected_parent_resume_sha,a.expected_parent_config_sha,a.expected_parent_protocol)!=(
            PARENT_RESUME_SHA,PARENT_CONFIG_SHA,PARENT_PROTOCOL):
        p.error('V4 accepts only the sealed best6500 parent')
    a.preflight_case=None
    return a


def main():
    a=parse();root=Path(a.root).resolve();repo=root/'repos/HYPIR_cuda'
    out=root/'adaptation/runs'/a.run
    parent_path=Path(a.parent_resume).resolve();parent_config=Path(a.parent_config).resolve()
    if out.resolve()==parent_path.parent or out.resolve()==parent_config.parent:
        raise RuntimeError('New stage must not write the parent run')
    if sha(parent_path)!=a.expected_parent_resume_sha or sha(parent_config)!=a.expected_parent_config_sha:
        raise RuntimeError('Sealed parent file SHA mismatch')
    parent_stat=(parent_path.stat().st_size,parent_path.stat().st_mtime_ns)
    parent_config_stat=(parent_config.stat().st_size,parent_config.stat().st_mtime_ns)
    cfg=json.loads(parent_config.read_text());parent_protocol=cfg['protocol']
    parent_id=hashlib.sha256(json.dumps(parent_protocol,sort_keys=True).encode()).hexdigest()
    if parent_id!=cfg['protocol_id'] or parent_id!=a.expected_parent_protocol:
        raise RuntimeError('Parent protocol mismatch')
    v1=root/'code/cuda_training/train_cuda_nr.py'
    v2=root/'code/cuda_training/train_cuda_nr_stage_v2.py'
    v3=root/'code/cuda_training/train_cuda_nr_stage_v3.py'
    if sha(v3)!=V3_SOURCE_SHA:raise RuntimeError('Frozen v3 derivation source changed')
    if sha(v2)!=V2_SOURCE_SHA:raise RuntimeError('Frozen v2 derivation source changed')
    if sha(v1)!=V1_SOURCE_SHA or parent_protocol['source_sha256'].get(str(v1))!=V1_SOURCE_SHA:
        raise RuntimeError('Frozen v1 helper source changed')
    for path,expected in parent_protocol['source_sha256'].items():
        if sha(path)!=expected:raise RuntimeError('Frozen parent source changed: '+path)
    expected={'schema':'cuda_raw_hypir_nr_stage_v3_parentanchor','base_step':6000,'coeff_t':175,'model_t':200,
              'rank':256,'batch':1,'patch':512,'lr':2e-7,'reward_weight':.075,
              'clip_weight':.1,'musiq_weight':1.5,'gradient_anchor':.3,'maniqa_train_samples':20,'nr_round_ste':False,
              'compile':False,'fuse':False}
    if any(parent_protocol[k]!=v for k,v in expected.items()):
        raise RuntimeError('This stage requires the sealed best6500 M1.5 objective')
    if cfg['target_stage_steps']!=250 or cfg['target_global_step']!=6500 or cfg['target_optimizer_step']!=500:
        raise RuntimeError('Parent must be completed250/global6500/Adam500')
    ancestor=parent_protocol['parent_lineage']
    teacher6250_path=Path(ancestor['parent_resume']).resolve()
    teacher6250_config=Path(ancestor['parent_config']).resolve()
    if (ancestor['parent_resume_sha256'],ancestor['parent_config_sha256'],ancestor['parent_protocol_id'])!=(
            TEACHER6250_RESUME_SHA,TEACHER6250_CONFIG_SHA,TEACHER6250_PROTOCOL):
        raise RuntimeError('Parent6250 lineage changed')
    teacher_path=parent_path if a.teacher=='parent6500' else teacher6250_path
    teacher_config=parent_config if a.teacher=='parent6500' else teacher6250_config
    teacher_sha=PARENT_RESUME_SHA if a.teacher=='parent6500' else TEACHER6250_RESUME_SHA
    teacher_config_sha=PARENT_CONFIG_SHA if a.teacher=='parent6500' else TEACHER6250_CONFIG_SHA
    teacher_protocol_id=PARENT_PROTOCOL if a.teacher=='parent6500' else TEACHER6250_PROTOCOL
    if sha(teacher_path)!=teacher_sha or sha(teacher_config)!=teacher_config_sha:
        raise RuntimeError('Sealed teacher file SHA mismatch')
    teacher_stat=(teacher_path.stat().st_size,teacher_path.stat().st_mtime_ns)
    teacher_cfg=json.loads(teacher_config.read_text())
    if teacher_cfg['protocol_id']!=teacher_protocol_id or hashlib.sha256(
            json.dumps(teacher_cfg['protocol'],sort_keys=True).encode()).hexdigest()!=teacher_protocol_id:
        raise RuntimeError('Teacher protocol mismatch')
    if out.resolve() in [teacher_path.parent,teacher_config.parent]:
        raise RuntimeError('New stage must not write its teacher run')
    spec=importlib.util.spec_from_file_location('_frozen_cuda_nr_v1_helpers',v1)
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    atomic_json,atomic_torch=helpers.atomic_json,helpers.atomic_torch
    seed_all,gradient_loss=helpers.seed_all,helpers.gradient_loss
    create_metrics=helpers.create_metrics
    NAMES=helpers.NAMES
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
    parent=torch.load(parent_path,map_location='cpu',weights_only=False,mmap=True)
    required={'protocol_id','lora','optimizer','local_step','global_step','python_rng','numpy_rng','torch_rng','cuda_rng'}
    if not required.issubset(parent) or parent['protocol_id']!=parent_id:
        raise RuntimeError('Parent state schema/protocol mismatch')
    if parent['local_step']!=250 or parent['global_step']!=6500 or parent['optimizer_step']!=500:
        raise RuntimeError('Parent resume must be local250/global6500/Adam500')
    states=parent['optimizer']['state'];groups=parent['optimizer']['param_groups']
    ids=[i for group in groups for i in group['params']]
    if len(states)!=514 or len(ids)!=514 or len(set(ids))!=514 or set(states)!=set(ids):
        raise RuntimeError('Parent Adam parameter IDs incomplete')
    if {int(v['step']) for v in states.values()}!={500}:
        raise RuntimeError('Parent Adam must be at step500')
    if len(parent['lora'])!=514 or sum(x.numel() for x in parent['lora'].values())!=259474432:
        raise RuntimeError('Parent LoRA514 tensor count/element count changed')
    if any(x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) for x in parent['lora'].values()):
        raise RuntimeError('Parent must hold finite FP32 LoRA, not a BF16 reloaded checkpoint')
    init=Path(parent_protocol['init_checkpoint']).resolve()
    if sha(init)!=INIT_SHA:raise RuntimeError('Original raw6000 initialization changed')
    teacher_saved=parent if a.teacher=='parent6500' else torch.load(
        teacher_path,map_location='cpu',weights_only=False,mmap=True)
    expected_teacher_global=6500 if a.teacher=='parent6500' else 6250
    if teacher_saved['protocol_id']!=teacher_protocol_id or teacher_saved['global_step']!=expected_teacher_global:
        raise RuntimeError('Teacher resume identity/step mismatch')
    if len(teacher_saved['lora'])!=514 or sum(x.numel() for x in teacher_saved['lora'].values())!=259474432:
        raise RuntimeError('Teacher LoRA514 tensor/element count changed')
    if any(x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) for x in teacher_saved['lora'].values()):
        raise RuntimeError('Teacher must hold finite exact FP32 LoRA')
    base=Path('/temp/zihaohe_stage/baselines/sd21b')
    provenance=parent_protocol['base_provenance']
    for records in [provenance['initial_core_asset_hashes'],provenance['all_config_tokenizer_scheduler_metadata_hashes']]:
        for rel,record in records.items():
            if sha(base/rel)!=record['sha256']:raise RuntimeError('Base asset changed: '+rel)
    input_records=parent_protocol['inputs']
    if len(input_records)!=100 or [x['case'] for x in input_records]!=list(range(1,101)):
        raise RuntimeError('Parent input sequence changed')
    for record in input_records:
        if sha(record['path'])!=record['sha256']:raise RuntimeError('Training input changed')
    for key in ['seed','coeff_t']:
        setattr(a,key,parent_protocol[key])
    a.maniqa_samples=parent_protocol['maniqa_train_samples']
    stage_start_global=6500;parent_optimizer_step=500
    objective_changes={k:{'parent':parent_protocol[k],'stage':getattr(a,k)}
                       for k in ['lr','reward_weight','clip_weight','musiq_weight','gradient_anchor']
                       if getattr(a,k)!=parent_protocol[k]}
    if a.teacher=='parent6500':objective_changes['anchor']={'parent':'parent6250','stage':'parent6500'}
    lineage={'parent_resume':str(parent_path),'parent_resume_sha256':a.expected_parent_resume_sha,
             'parent_config':str(parent_config),'parent_config_sha256':a.expected_parent_config_sha,
             'parent_protocol_id':parent_id,'parent_local_step':250,'parent_global_step':6500,
             'parent_optimizer_step':500,'frozen_v1_source':str(v1),'frozen_v1_source_sha256':V1_SOURCE_SHA,
             'restore_order':'raw6000 BF16 init -> LoRA.float -> exact parent FP32 copy -> parent AdamW -> all4 parent RNG',
             'objective_relation':'sealed best6500 exact-FP32 continuation; inherited Adam moments/step and all4 RNG; objective changes explicit',
             'objective_changes':objective_changes,'teacher':a.teacher,'teacher_resume':str(teacher_path),
             'teacher_resume_sha256':teacher_sha,'teacher_config':str(teacher_config),
             'teacher_config_sha256':teacher_config_sha,'teacher_protocol_id':teacher_protocol_id,
             'musiq_weight':a.musiq_weight,'derived_v2_source':str(v2),'derived_v2_source_sha256':V2_SOURCE_SHA,
             'derived_v3_source':str(v3),'derived_v3_source_sha256':V3_SOURCE_SHA,
             'optimizer_warmstart':'inherit all514 FP32 Adam moments and step500, then explicitly set all group lr to stage lr',
             'parent_lr':parent_protocol['lr'],'stage_lr':a.lr,'lr_overwrite_after_load':True,
             'lr_changed':a.lr!=parent_protocol['lr']}
    if a.resume:
        if not out.is_dir() or Path(a.resume).resolve()!=(out/'resume_latest.pt').resolve():
            raise RuntimeError('Stage resume must explicitly select its own resume_latest.pt')
    else:out.mkdir(parents=True,exist_ok=False)
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_file=out/'resume_latest.pt'
        seed_all(a.seed,torch,np)
        def enhancer(weight):
            m=SD2Enhancer(base_model_path=str(base),weight_path=str(weight),lora_modules=helpers.MODULES.split(','),
                          lora_rank=256,model_t=200,coeff_t=a.coeff_t,device=str(device))
            m.init_models();m.G.eval().requires_grad_(False);m.vae.eval().requires_grad_(False)
            m.text_encoder.eval().requires_grad_(False);m.prepare_inputs(1,'')
            for name,p in m.G.named_parameters():
                if 'lora' in name:p.data=p.data.float()
            return m
        student=enhancer(init);teacher=enhancer(init)
        # Never load the resume as a weight file: copy its exact FP32 LoRA.
        teacher_current=teacher.G.state_dict()
        if set(teacher_saved['lora'])!={k for k in teacher_current if 'lora' in k}:
            raise RuntimeError('Teacher parent LoRA keys changed')
        with torch.no_grad():
            for k,v in teacher_saved['lora'].items():
                if teacher_current[k].dtype!=torch.float32 or v.dtype!=torch.float32:
                    raise RuntimeError('Teacher restoration would quantize parent FP32')
                teacher_current[k].copy_(v.to(teacher_current[k]))
        teacher_exact=all(torch.equal(teacher_current[k].detach().cpu(),v) for k,v in teacher_saved['lora'].items())
        if not teacher_exact or any(p.requires_grad for module in [teacher.G,teacher.vae,teacher.text_encoder]
                                    for p in module.parameters()):
            raise RuntimeError('Frozen teacher does not exactly equal selected resume FP32')
        del teacher_current
        named=[(name,p) for name,p in student.G.named_parameters() if 'lora' in name]
        params=[p for _,p in named]
        for p in params:p.requires_grad_(True)
        if len(params)!=514 or [k for k,_ in named]!=list(parent['lora']):
            raise RuntimeError('Parent optimizer/LoRA parameter order changed')
        for pid,(name,p) in zip(ids,named):
            st=states[pid]
            if st['exp_avg'].shape!=p.shape or st['exp_avg_sq'].shape!=p.shape:
                raise RuntimeError('Parent Adam shape does not match LoRA order')
            if st['exp_avg'].dtype!=torch.float32 or st['exp_avg_sq'].dtype!=torch.float32:
                raise RuntimeError('Parent Adam moments must remain FP32')
            if not bool(torch.isfinite(st['exp_avg']).all()) or not bool(torch.isfinite(st['exp_avg_sq']).all()):
                raise RuntimeError('Parent Adam moments must be finite')
        student.G.enable_gradient_checkpointing()
        metrics,clip,maniqa_proof,metric_sha=create_metrics(root,device,a.maniqa_samples)
        if metric_sha!=parent_protocol['metric_weights_sha256']:
            raise RuntimeError('Metric weights changed from parent')
        versions={name:importlib.metadata.version(name) for name in parent_protocol['versions']}
        if versions!=parent_protocol['versions']:raise RuntimeError('Training runtime versions changed')
        source_sha=dict(parent_protocol['source_sha256']);source_sha[str(Path(__file__).resolve())]=sha(__file__)
        source_sha[str(v2)]=V2_SOURCE_SHA
        source_sha[str(v3)]=V3_SOURCE_SHA
        protocol=dict(parent_protocol)
        # Remove inherited assertions which belong specifically to the v3 teacher.
        protocol.pop('teacher_FP32_exact_parent6250',None)
        protocol.update(schema='cuda_raw_hypir_nr_stage_v4_best6500',parent_lineage=lineage,
            anchor_checkpoint=str(teacher_path),anchor_sha256=teacher_sha,
            anchor_initialization_checkpoint=str(init),anchor_initialization_sha256=INIT_SHA,
            anchor_loading='all514 selected sealed resume.lora FP32 values copied exactly into raw6000-initialized teacher; never direct resume-file weight loading',
            anchor=f'frozen {a.teacher} exact FP32 LoRA from sealed resume, same latent/coeff175',
            teacher=a.teacher,teacher_FP32_exact_sealed_resume=True,teacher_frozen=True,musiq_weight=a.musiq_weight,
            lr=a.lr,reward_weight=a.reward_weight,clip_weight=a.clip_weight,gradient_anchor=a.gradient_anchor,
            fidelity_L1_weight=1.,fidelity_LPIPS_weight=.1,
            fresh_stage_initial_fidelity_expected_zero=(a.teacher=='parent6500'),initial_fidelity_tolerance=1e-6,
            loss='L1(selected_teacher)+.1*LPIPS(selected_teacher)+gradient_anchor*spatial_gradient_L1 - reward_weight*(musiq_weight*MUSIQ/100+MANIQA+clip_weight*CLIPIQA)',
            stage_start_global_step=stage_start_global,parent_optimizer_step=parent_optimizer_step,
            local_step_semantics='this new stage only; global6500+local; Adam500+local',
            optimizer='AdamW inherited parent FP32 parameters/moments/step/options; all group lr explicitly overwritten to configured lr after load',
            nr_round_ste=a.nr_round_ste,
            quantization='unencoded RGB; no quantization surrogate',
            checkpoint_policy='raw step/global checkpoint and exact Adam/RNG resume every250 and final; milestone resume hardlinked from atomic latest inode',
            source_sha256=source_sha,versions=versions)
        identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0,foreach=False)
        seed_all(a.seed,torch,np);done=0;saved=parent
        if a.resume:
            prior_config=json.loads((out/'config.json').read_text())
            if prior_config['target_stage_steps']!=a.stage_steps or prior_config['save_every']!=a.save_every:
                raise RuntimeError('Stage resume target steps/checkpoint policy changed')
            saved=torch.load(Path(a.resume),map_location='cpu',weights_only=False,mmap=True)
            if saved['protocol_id']!=identity:raise RuntimeError('Stage resume protocol changed')
            done=saved['local_step']
            if not 0<=done<=a.stage_steps or saved['global_step']!=stage_start_global+done:
                raise RuntimeError('Stage resume step mismatch')
        current=student.G.state_dict()
        if set(saved['lora'])!={k for k in current if 'lora' in k}:
            raise RuntimeError('Restored LoRA key mismatch')
        with torch.no_grad():
            for k,v in saved['lora'].items():
                if v.dtype!=torch.float32 or current[k].dtype!=torch.float32:
                    raise RuntimeError('FP32 restoration would quantize LoRA')
                current[k].copy_(v.to(current[k]))
        # Equality proof occurs before any RNG restoration or optimizer update.
        if not all(torch.equal(current[k].detach().cpu(),v) for k,v in saved['lora'].items()):
            raise RuntimeError('Restored FP32 LoRA differs from saved values')
        opt.load_state_dict(saved['optimizer'])
        if {int(v['step']) for v in opt.state.values()}!={parent_optimizer_step+done}:
            raise RuntimeError('Restored Adam step mismatch')
        saved_ids=[pid for group in saved['optimizer']['param_groups'] for pid in group['params']]
        if len(saved_ids)!=514:raise RuntimeError('Restored Adam parameter count changed')
        moment_exact=all(
            opt.state[p][key].dtype==torch.float32 and
            torch.equal(opt.state[p][key].detach().cpu(),saved['optimizer']['state'][pid][key])
            for p,pid in zip(params,saved_ids) for key in ['exp_avg','exp_avg_sq'])
        if not moment_exact:raise RuntimeError('Restored FP32 Adam moments differ from saved values')
        loaded_lrs=[group['lr'] for group in opt.param_groups]
        expected_loaded_lr=a.lr if a.resume else parent_protocol['lr']
        if any(value!=expected_loaded_lr for value in loaded_lrs):
            raise RuntimeError('Restored Adam group lr differs from sealed source stage')
        for group in opt.param_groups:group['lr']=a.lr
        if any(group['lr']!=a.lr for group in opt.param_groups):raise RuntimeError('Configured lr overwrite failed')
        atomic_json(out/'config.json',dict(protocol_id=identity,protocol=protocol,target_stage_steps=a.stage_steps,
            target_global_step=stage_start_global+a.stage_steps,target_optimizer_step=parent_optimizer_step+a.stage_steps,
            save_every=a.save_every))
        atomic_json(out/'parent_migration.json',dict(protocol_id=identity,parent_lineage=lineage,
            restored_from=str(a.resume or parent_path),restored_stage_local_step=done,
            restored_global_step=stage_start_global+done,restored_optimizer_step=parent_optimizer_step+done,
            FP32_LoRA_exact_equal_before_update=True,FP32_LoRA_tensors=514,
            FP32_Adam_moments_exact_equal_before_update=moment_exact,Adam_parameter_states=514,
            teacher=a.teacher,teacher_FP32_exact_sealed_resume=teacher_exact,teacher_FP32_tensors=514,teacher_frozen=True,
            anchor_resume_sha256=teacher_sha,musiq_weight=a.musiq_weight,
            loaded_optimizer_group_lrs=loaded_lrs,applied_optimizer_group_lrs=[g['lr'] for g in opt.param_groups],
            lr_overwrite_after_load=True,objective_changes=objective_changes,
            source_sha256=source_sha,snapshot_phase='before_new_stage_updates',
            optimizer_updates_in_new_stage=done))
        atomic_json(out/'maniqa_initial_state_proof.json',maniqa_proof)
        if (parent_path.stat().st_size,parent_path.stat().st_mtime_ns)!=parent_stat:
            raise RuntimeError('Parent resume changed during initialization')
        if (parent_config.stat().st_size,parent_config.stat().st_mtime_ns)!=parent_config_stat:
            raise RuntimeError('Parent config changed during initialization')
        if (teacher_path.stat().st_size,teacher_path.stat().st_mtime_ns)!=teacher_stat:
            raise RuntimeError('Teacher resume changed during initialization')
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'],0)
        del saved,parent,teacher_saved,states,current
        begin=time.time();first=True
        with (out/'train.jsonl').open('a',buffering=1) as log:
            for local_step in range(done+1,a.stage_steps+1):
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
                metric_rgb=rgb
                with torch.autocast('cuda',enabled=False):
                    lpips=metrics['lpips'](rgb,target).mean()
                    fidelity=(rgb-target).abs().mean()+.1*lpips+a.gradient_anchor*gradient_loss(rgb,target)
                    scores={n:metrics[n](metric_rgb).mean()/(100 if n=='musiq' else 1) for n in NAMES[:3]}
                    reward=a.musiq_weight*scores['musiq']+scores['maniqa']+a.clip_weight*scores['clipiqa'];loss=fidelity-a.reward_weight*reward
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                if first and done==0 and a.teacher=='parent6500' and abs(float(fidelity.detach()))>1e-6:
                    raise RuntimeError('Fresh student/teacher6500 fidelity should be numerically zero')
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
                finite_lora_gradients=sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
                if not torch.isfinite(norm) or nonzero!=514 or finite_lora_gradients!=514:
                    raise RuntimeError('All514 LoRA gradients must be finite and nonzero')
                if any(p.grad is not None for p in teacher.G.parameters()):raise RuntimeError('Frozen anchor got parameter gradient')
                if any(p.grad is not None for m in list(metrics.values())+[clip] for p in m.parameters()):
                    raise RuntimeError('Frozen metric got parameter gradient')
                torch.cuda.synchronize();row=dict(local_step=local_step,global_step=stage_start_global+local_step,
                    optimizer_step=parent_optimizer_step+local_step,case=case,crop_xy=[left,top],
                    loss=float(loss.detach()),fidelity=float(fidelity.detach()),reward=float(reward.detach()),
                    scores={k:float(v.detach()) for k,v in scores.items()},gradient_norm=float(norm),
                    nonzero_lora_gradients=nonzero,finite_lora_gradients=finite_lora_gradients,
                    lr=opt.param_groups[0]['lr'],teacher=a.teacher,
                    step_seconds=time.time()-start,elapsed_seconds=time.time()-begin,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                if first:
                    atomic_json(out/'gradient_preflight.json',dict(status='passed',components=audit,row=row,
                                no_optimizer_update_yet=True,frozen_metrics=True,frozen_same_latent_anchor=True,
                                teacher=a.teacher,teacher_FP32_exact_sealed_resume=True,
                                initial_fidelity_expected_zero=(done==0 and a.teacher=='parent6500'),
                                initial_fidelity_tolerance=1e-6,musiq_weight=a.musiq_weight))
                    first=False
                if a.preflight_only:
                    atomic_json(out/'status.json',dict(stage='preflight_complete',row=row,optimizer_updates=0));print(json.dumps(row),flush=True);return
                opt.step();log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                atomic_json(out/'status.json',dict(stage='training',completed_local_steps=local_step,total=a.stage_steps,row=row))
                if local_step%a.save_every==0 or local_step==a.stage_steps:
                    lora={k:v.detach().cpu().clone() for k,v in student.G.state_dict().items() if 'lora' in k}
                    atomic_torch(out/f'step{stage_start_global+local_step}.pth',lora,torch)
                    state=dict(protocol_id=identity,lora=lora,optimizer=opt.state_dict(),local_step=local_step,
                               global_step=stage_start_global+local_step,optimizer_step=parent_optimizer_step+local_step,
                               parent_lineage=lineage,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                               torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0))
                    atomic_torch(state_file,state,torch)
                    milestone=out/f'resume_step{stage_start_global+local_step}.pt'
                    # atomic_torch replaced latest's inode; a hardlink preserves this
                    # exact immutable milestone while future saves replace latest.
                    if milestone.exists():raise RuntimeError('Milestone resume already exists')
                    os.link(state_file,milestone)
                    if milestone.stat().st_ino!=state_file.stat().st_ino:
                        raise RuntimeError('Milestone resume hardlink failed')
                    record=dict(global_step=stage_start_global+local_step,local_step=local_step,
                        optimizer_step=parent_optimizer_step+local_step,
                        raw_checkpoint=str(out/f'step{stage_start_global+local_step}.pth'),
                        raw_checkpoint_sha256=sha(out/f'step{stage_start_global+local_step}.pth'),
                        resume_checkpoint=str(milestone),resume_checkpoint_sha256=sha(milestone),
                        resume_bytes=milestone.stat().st_size,resume_mtime_ns=milestone.stat().st_mtime_ns,
                        resume_inode=milestone.stat().st_ino,latest_same_inode_at_save=True,
                        exact_FP32_LoRA_Adam_and_all4_RNG=True)
                    atomic_json(out/f'checkpoint_step{stage_start_global+local_step}.json',record)
                    del state,lora
                del x,z,pred,rgb,metric_rgb,target,loss,reward,scores,lpips,fidelity
        if (parent_path.stat().st_size,parent_path.stat().st_mtime_ns)!=parent_stat or (
            parent_config.stat().st_size,parent_config.stat().st_mtime_ns)!=parent_config_stat or (
            teacher_path.stat().st_size,teacher_path.stat().st_mtime_ns)!=teacher_stat:
            raise RuntimeError('Sealed parent/teacher changed during training')
        for path,expected in source_sha.items():
            if sha(path)!=expected:raise RuntimeError('Frozen source changed during training: '+path)
        atomic_json(out/'status.json',dict(stage='complete',completed_local_steps=a.stage_steps,
            global_step=stage_start_global+a.stage_steps,optimizer_step=parent_optimizer_step+a.stage_steps,
            requires_independent_quality_evaluation=True))
        (out/f'training_complete_step{stage_start_global+a.stage_steps}').touch()


if __name__=='__main__':
    main()
