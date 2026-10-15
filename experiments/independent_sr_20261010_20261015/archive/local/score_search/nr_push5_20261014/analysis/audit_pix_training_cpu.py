"""Independent CPU proof for the actual0/125/250 pixel-student snapshots."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import pickle
import random

os.environ['CUDA_VISIBLE_DEVICES']=''
TRAIN_SHA='3a389e55e898be8a64cdda7637339126b1caf94d0b9e7fb19c9f66abe61e8a16'
CLI_SHA='d0b1d8560b9bdf43e628fe6fa0b8191b24a85cadc4c4f27fe9964823a7cd4b1c'
G4_SHA='283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83'
TEACHER_SHA='35355c94cace390b5f5918cfb607d85b0f55c9cd1e1ba0a4f6a4d034abb54e16'


def need(ok,message):
    if not ok:raise RuntimeError(message)


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--step',type=int,choices=[0,125,250],required=True);args=ap.parse_args();R=args.root;n=args.step
    d=R/'adaptation/runs/pix_push5_g4_nr250';raw=d/f'step{n}.pth';resume=d/f'resume_step{n}.pt'
    marker=read(d/f'checkpoint_step{n}.json');config=read(d/'config.json');p=config['protocol']
    need(sha(R/'code/push5/train_pix_g4_nr_stage1.py')==TRAIN_SHA and
         sha(R/'code/push5/g4_stage1_conservative250.json')==CLI_SHA,'Actual frozen trainer/CLI changed')
    need(hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()==config['protocol_id'],'Protocol digest differs')
    need(p['schema']=='pix_g4_A7750_offline_distill_nr_stage1_v1' and config['initial_optimizer_step']==0 and
         config['target_stage_steps']==250 and config['save_every']==125,'New-stage configuration differs')
    need(p['initial_reference_sha256']==G4_SHA and p['teacher_manifest_sha256']==TEACHER_SHA and
         p['lr']==1e-5 and p['reward_weight']==.01 and p['musiq_weight']==1 and p['clip_weight']==.5 and
         p['gradient_anchor']==.3 and p['fidelity_L1_weight']==1 and p['fidelity_LPIPS_weight']==.1,'Objective differs')
    for path,digest in p['source_sha256'].items():need(sha(path)==digest,'Training source changed')
    need([x['case'] for x in p['inputs']]==list(range(1,101)) and
         all(Path(x['path']).resolve().parent==R/'source_payload/assets/input' for x in p['inputs']),
         'Training inputs not exactly test100')
    need(sha(p['teacher_manifest'])==TEACHER_SHA,'Offline teacher manifest changed')
    catalog=[x for x in read(p['teacher_manifest'])['records'] if x['split']=='test']
    teacher=p['teacher_records'];need([x['case'] for x in teacher]==list(range(1,101)),'Teacher test100 order differs')
    for src,rec,cat in zip(p['inputs'],teacher,catalog):
        need(rec['case']==src['case']==cat['case'] and rec['sha256']==cat['sha256'] and
             Path(rec['path']).name==f"case{rec['case']}.png" and sha(rec['path'])==rec['sha256'],
             'Actual teacher PNG/case SHA differs')
    need(Path(marker['raw_checkpoint'])==raw and Path(marker['resume_checkpoint'])==resume and
         marker['local_step']==marker['optimizer_step']==n and sha(raw)==marker['raw_sha256'] and
         sha(resume)==marker['resume_sha256'],'Immutable marker identity differs')
    s=resume.stat();need((s.st_size,s.st_mtime_ns,s.st_ino)==
        (marker['resume_bytes'],marker['resume_mtime_ns'],marker['resume_inode']),'Immutable resume stat changed')
    if n==250:
        status=read(d/'status.json');need(status['stage']=='complete' and status['completed_local_steps']==250 and
            (d/'training_complete_step250').is_file(),'Terminal training not complete')
    import torch
    import numpy as np
    need(not torch.cuda.is_initialized(),'CUDA initialized by CPU audit');torch.set_num_threads(2)
    original=torch.load(R/'calibration/g4/model_dir/model.pth',map_location='cpu',weights_only=False)
    need(sha(R/'calibration/g4/model_dir/model.pth')==G4_SHA,'Original G4 checkpoint differs')
    rr=torch.load(raw,map_location='cpu',weights_only=False);ss=torch.load(resume,map_location='cpu',weights_only=False)
    init=original['state'];state=rr['state'];need(len(state)==len(init)==len(ss['state'])==100 and
        list(state)==list(init)==list(ss['state'])==p['trainable_parameter_names'],'Actual state100 order differs')
    need(rr['step']==ss['local_step']==ss['optimizer_step']==n and rr['reference_step']==30000 and
         rr['protocol_id']==ss['protocol_id']==config['protocol_id'],'Saved step/protocol differs')
    need(rr['cfg']==ss['cfg']==original['cfg'],'Saved architecture differs')
    changed=0;l2=0.;maxabs=0.
    for key,v in state.items():
        need(v.dtype==ss['state'][key].dtype==torch.float32 and v.shape==init[key].shape and
             torch.isfinite(v).all().item() and torch.equal(v,ss['state'][key]),'Raw/resume exact FP32 state differs')
        delta=(v.double()-init[key].double());changed+=not torch.equal(v,init[key]);l2+=delta.square().sum().item()
        maxabs=max(maxabs,delta.abs().max().item())
    need(changed==(0 if n==0 else 100),'Expected actual parameter changes absent')
    opt=ss['optimizer'];ids=[i for g in opt['param_groups'] for i in g['params']]
    need(len(ids)==len(set(ids))==100 and opt['param_groups'][0]['lr']==1e-5 and
         opt['param_groups'][0]['weight_decay']==0 and opt['param_groups'][0]['foreach'] is False,'Adam group differs')
    need(len(opt['state'])==(0 if n==0 else 100),'New Adam state count differs')
    if n:
        need(set(opt['state'])==set(ids),'Adam parameter IDs differ')
        for pid,v in zip(ids,state.values()):
            item=opt['state'][pid];need(float(item['step'])==n,'Actual Adam step differs')
            for name in ['exp_avg','exp_avg_sq']:
                m=item[name];need(m.dtype==torch.float32 and m.shape==v.shape and torch.isfinite(m).all().item(),'Adam moment invalid')
            need((item['exp_avg_sq']>=0).all().item(),'Adam second moment negative')
    rng={};q=random.Random(0);q.setstate(ss['python_rng']);np.random.RandomState(0).set_state(ss['numpy_rng'])
    for key in ['torch_rng','cuda_rng']:need(ss[key].dtype==torch.uint8 and ss[key].ndim==1 and ss[key].numel()>0,'RNG tensor invalid')
    for key in ['python_rng','numpy_rng','torch_rng','cuda_rng']:
        rng[key]=hashlib.sha256(ss[key].numpy().tobytes() if torch.is_tensor(ss[key]) else pickle.dumps(ss[key],protocol=4)).hexdigest()
    g=torch.Generator();g.set_state(ss['torch_rng'])
    if n==0:
        need(ss['python_rng']==random.Random(p['seed']).getstate() and
             torch.equal(ss['torch_rng'],torch.Generator().manual_seed(p['seed']).get_state()),'New initial RNG differs')
    rows=[json.loads(x) for x in (d/'train.jsonl').read_text().splitlines() if x.strip()][:n]
    need(len(rows)==n,'Training log too short')
    for i,row in enumerate(rows,1):
        need(row['local_step']==row['optimizer_step']==i and row['lr']==1e-5 and
             row['finite_parameter_gradients']==row['nonzero_parameter_gradients']==100,'Actual update/gradient row differs')
        case=row['case'];rec=teacher[case-1];x,y=row['context_crop_xy'];w,h=rec['size']
        need(1<=case<=100 and x%8==y%8==0 and 0<=x<=w-1024 and 0<=y<=h-1024 and
             row['loss_crop_xy']==[x+256,y+256] and row['teacher_png']==rec['path'] and
             row['teacher_png_sha256']==rec['sha256'],'Case/teacher/native crop alignment differs')
        fid=row['L1']+.1*row['LPIPS']+.3*row['spatial_gradient_L1']
        rew=row['scores']['musiq']+row['scores']['maniqa']+.5*row['scores']['clipiqa']
        need(all(math.isfinite(row[k]) for k in ['loss','fidelity','reward','gradient_norm']) and
             abs(fid-row['fidelity'])<1e-6 and abs(rew-row['reward'])<1e-6 and
             abs(row['loss']-(fid-.01*rew))<1e-6,'Actual logged loss/reward differs')
    first=read(d/'gradient_preflight.json');need(first['status']=='PASS' and first['no_optimizer_update_yet'] and
        first['frozen_metrics_and_offline_teacher'] and all(x['finite'] and x['nonzero'] and x['input_gradient_norm']>0 for x in first['components'].values()),
        'Real frozen-metric input gradients not proven')
    fr=first['row'];q=random.Random(p['seed']);case=q.randint(1,100);w,h=teacher[case-1]['size']
    crop=[8*q.randrange((w-1024)//8+1),8*q.randrange((h-1024)//8+1)]
    need(fr['case']==case and fr['context_crop_xy']==crop,'Independent first crop replay differs')
    need(not torch.cuda.is_initialized(),'CUDA initialized at end')
    print(json.dumps({'schema':'push5_independent_pixel_training_snapshot_CPU_v1','status':'PASS','CPU_only':True,
        'CUDA_context_initialized':False,'run':d.name,'local_step':n,'optimizer_step':n,'reference_training_step':30000,
        'state_tensors':100,'FP32_parameters':sum(v.numel() for v in state.values()),'all_finite':True,
        'raw_exact_equal_resume_FP32':True,'changed_state_tensors_vs_G4':changed,'parameter_delta_L2':math.sqrt(l2),
        'parameter_delta_max_abs':maxabs,'Adam_state_entries':len(opt['state']),'Adam_moments_FP32_finite':True,
        'four_RNG_state_SHA256':rng,'actual100_offline_teacher_PNG_SHA_PASS':True,'teacher_catalog_sha256':TEACHER_SHA,
        'same_case_crop_alignment_and_loss_audited_updates':n,'first_case':case,'first_context_crop_xy':crop,
        'original_reference_checkpoint_sha256':G4_SHA,'source_sha256':TRAIN_SHA,'CLI_sha256':CLI_SHA,
        'raw_sha256':marker['raw_sha256'],'resume_sha256':marker['resume_sha256'],
        'protocol_id':config['protocol_id'],'scope':'Actual saved state and logged training controls only; no full metric quality claim.'},indent=2))


if __name__=='__main__':main()
