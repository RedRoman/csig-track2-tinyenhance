"""Independent CPU artifact/log/RNG audit, no model or CUDA construction."""
from pathlib import Path
import argparse, hashlib, json, math, os, random, statistics
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/cache/independent_sr_cuda_20261013')
SRC_SHA='8a90192c200efa934e3234984855575c1b25feffdc1279cd64e3db10619f1c4c'
A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'

def need(v,s):
    if not v:raise RuntimeError(s)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def rec(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='hypir_push7_A7750_direct_nr100');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    need(not a.out.exists(),'New audit output required')
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CPU only')
    d=R/'adaptation/runs'/a.run;c=json.loads((d/'config.json').read_text());p=c['protocol'];obj=p['objective']
    pid=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();need(pid==c['protocol_id'],'Protocol hash')
    need(p['parent_raw_sha256']==A_SHA and p['new_Adam_initial_step']==0 and p['stage_steps']==100,'Initial A/freshAdam/target')
    need(p['teacher_output_loss'] is False and p['distillation'] is False and p['validation_GT_training'] is False,'No teacher/GT scope')
    source=R/'code/push7/hypir_direct_nr/train_hypir_direct_nr_v2.py';need(sha(source)==SRC_SHA,'Frozen trainer')
    for path,h in p['source_sha256'].items():need(sha(path)==h,'Frozen source '+path)
    inputs=json.loads((d/'input_catalog.json').read_text());need(inputs==p['inputs'] and [x['case'] for x in inputs]==list(range(1,101)),'Test100 catalog')
    sizes={}
    for r in inputs:
        path=R/'source_payload/assets/input'/f"case{r['case']}.jpg"
        need(Path(r['path']).resolve()==path and sha(path)==r['sha256'],'Actual test input')
        with Image.open(path) as im:sizes[r['case']]=im.size
        need(not({'teacher','gt','target'}&set(r)),'No teacher/GT entries')
    rows=[json.loads(x) for x in (d/'train.jsonl').read_text().splitlines() if x.strip()]
    need([r['local_step'] for r in rows]==list(range(1,101)),'Exactly100 real update rows, no duplicates/gaps')
    for i,r in enumerate(rows,1):
        need(r['global_step']==7750+i and r['optimizer_step']==i and r['lr']==obj['lr'],'Actual counters/LR')
        need(r['finite_lora_gradients']==r['nonzero_lora_gradients']==514,'All514 actual gradients')
        numeric=[r[n] for n in ['loss','content','L1_input','LPIPS_input','gradient_input','parameter_anchor','reward','gradient_norm','update_RMS_from_initial','update_relative_L2_from_initial','update_max_tensor_relative_L2','update_mean_tensor_relative_squared']]+list(r['scores'].values())
        need(all(math.isfinite(x) for x in numeric),'All log values finite')
        content=obj['input_l1']*r['L1_input']+obj['input_lpips']*r['LPIPS_input']+obj['input_gradient']*r['gradient_input']
        reward=obj['musiq_weight']*r['scores']['musiq']+r['scores']['maniqa']+obj['clip_weight']*r['scores']['clipiqa']
        need(abs(content-r['content'])<3e-7 and abs(reward-r['reward'])<5e-7,'Independent component recompute')
        need(abs(content+obj['parameter_anchor']*r['parameter_anchor']-obj['reward_weight']*reward-r['loss'])<5e-7,'Independent loss recompute')
        if i>1:need(abs(r['parameter_anchor']-rows[i-2]['update_mean_tensor_relative_squared'])<1e-9,'Parameter anchor actual previous update')
    files={};milestones={};states={};initial=torch.load(Path(p['parent_raw']),map_location='cpu',weights_only=True,mmap=True)
    need(sha(p['parent_raw'])==A_SHA and len(initial)==514,'Actual initial A')
    saved_locals=[0]+([1] if (d/'checkpoint_step7751.json').exists() else [])+[50,100]
    for local in saved_locals:
        glob=7750+local;mp=d/f'checkpoint_step{glob}.json';m=json.loads(mp.read_text());sp=Path(m['resume_checkpoint']);rp=Path(m['raw_checkpoint'])
        need((m['global_step'],m['local_step'],m['optimizer_step'])==(glob,local,local) and m['protocol_id']==pid,'Marker identity/counters')
        need(m['no_teacher_outputs'] is True and m['no_val_GT'] is True and m['source_sha256']==p['source_sha256'],'Actual provenance')
        need(sha(sp)==m['resume_checkpoint_sha256'] and sha(rp)==m['raw_checkpoint_sha256'],'Actual immutable SHA')
        saved=torch.load(sp,map_location='cpu',weights_only=False,mmap=True);raw=torch.load(rp,map_location='cpu',weights_only=True,mmap=True)
        need((saved['global_step'],saved['local_step'],saved['optimizer_step'])==(glob,local,local) and saved['protocol_id']==pid,'Saved counters')
        need(list(raw)==list(saved['lora'])==list(initial) and len(raw)==514 and sum(v.numel() for v in raw.values())==259474432,'Original514 key order/size')
        need(all(v.dtype==torch.float32 and torch.isfinite(v).all() and torch.equal(v,saved['lora'][k]) for k,v in raw.items()),'Raw exact FP32 = resume')
        opt=saved['optimizer'];ids=[i for g in opt['param_groups'] for i in g['params']]
        need(len(ids)==len(set(ids))==514 and all(g['lr']==obj['lr'] and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 for g in opt['param_groups']),'Fresh Adam options')
        if not local:need(not opt['state'] and all(torch.equal(v,initial[k]) for k,v in raw.items()),'ExactA/FreshAdam0')
        else:
            need(set(ids)==set(opt['state']) and {int(v['step']) for v in opt['state'].values()}=={local},'All514 Adam steps')
            for (name,value),ident in zip(raw.items(),ids):
                st=opt['state'][ident]
                need(all(st[k].dtype==torch.float32 and st[k].device.type=='cpu' and st[k].shape==value.shape and torch.isfinite(st[k]).all() for k in ['exp_avg','exp_avg_sq']),'Adam moment dtype/shape/finiteness')
        changed=sum(not torch.equal(v,initial[k]) for k,v in raw.items());need(changed==m['changed_FP32_LoRA_tensors']==(514 if local else 0),'Actual514 weight changes')
        need(all(k in saved for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']),'Four real RNG saved')
        for key in ['torch_rng','cuda_rng']:need(saved[key].device.type=='cpu' and saved[key].dtype==torch.uint8 and saved[key].ndim==1 and saved[key].numel()>=16,'Actual RNG ByteTensor')
        states[local]={k:saved[k] for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']}
        milestones[str(local)]=dict(global_step=glob,optimizer_step=local,FP32_tensors=514,FP32_elements=259474432,raw_exact_equal_resume=True,actual_changed_tensors=changed,Adam_states=len(opt['state']),Adam_moments_FP32_CPU_finite=True,raw_sha256=m['raw_checkpoint_sha256'],resume_sha256=m['resume_checkpoint_sha256'])
        files[str(mp)]=rec(mp);files[str(sp)]=dict(path=str(sp),sha256=m['resume_checkpoint_sha256'],bytes=sp.stat().st_size);files[str(rp)]=dict(path=str(rp),sha256=m['raw_checkpoint_sha256'],bytes=rp.stat().st_size)
        if local==100:
            delta2=[(v.double()-initial[k].double()).square().sum().item() for k,v in raw.items()];den=[v.double().square().sum().item() for v in initial.values()]
            rms=math.sqrt(sum(delta2)/259474432);relative=math.sqrt(sum(delta2)/sum(den));relative_max=math.sqrt(max(v/max(n,1e-12) for v,n in zip(delta2,den)))
            need(abs(rms-rows[-1]['update_RMS_from_initial'])<1e-10 and abs(relative-rows[-1]['update_relative_L2_from_initial'])<1e-7 and abs(relative_max-rows[-1]['update_max_tensor_relative_L2'])<1e-7,'Independent final actual update recompute')
        del saved,raw,opt
    py=random.Random();py.setstate(states[0]['python_rng']);nr=np.random.RandomState();nr.set_state(states[0]['numpy_rng'])
    def np_equal(a,b):return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
    for step,r in enumerate(rows,1):
        case=py.randint(1,100);w,h=sizes[case];crop=[py.randrange(w-511),py.randrange(h-511)]
        need((case,crop)==(r['case'],r['crop_xy']),'Independently replayed case/crop RNG')
        for _ in range(20):nr.randint(0,512-224+1);nr.randint(0,512-224+1)
        if step in states:
            need(py.getstate()==states[step]['python_rng'],'Python RNG exact milestone')
            need(np_equal(nr.get_state(),states[step]['numpy_rng']),'MANIQA20 NumPy RNG exact milestone')
    need(all(torch.equal(states[0]['torch_rng'],s['torch_rng']) for s in states.values()),'CPU Torch RNG unchanged as deterministic source requires')
    offsets={k:int.from_bytes(v['cuda_rng'][8:16].numpy().tobytes(),'little') for k,v in states.items()};seeds={int.from_bytes(v['cuda_rng'][:8].numpy().tobytes(),'little') for v in states.values()}
    first_saved=min(k for k in offsets if k);need((offsets[first_saved]-offsets[0])%first_saved==0,'CUDA integer counter increment')
    increment=(offsets[first_saved]-offsets[0])//first_saved;need(len(seeds)==1 and increment>0 and all(offsets[k]==offsets[0]+increment*k for k in offsets),'CUDA saved Philox seed/counter actual per-step continuity')
    status=json.loads((d/'status.json').read_text());need(status['stage']=='complete' and status['completed_local_steps']==status['optimizer_step']==100,'Actual terminal status')
    for name in ['config.json','input_catalog.json','train.jsonl','status.json','initialization_proof.json','gradient_preflight.json']:files[str(d/name)]=rec(d/name)
    need(not torch.cuda.is_initialized(),'No CUDA context')
    result=dict(schema='hypir_push7_independent_training_cpu_v1',status='PASS',CPU_only=True,CUDA_initialized=False,run=str(d),protocol_id=pid,training_source_sha256=SRC_SHA,initial_A_sha256=A_SHA,no_teacher_outputs=True,no_val_GT_training=True,new_Adam_initial_step=0,actual_training_updates=100,all100_logs_finite_and_counters_and_loss_recomputed=True,actual_test_inputs_SHA_checked=100,milestones=milestones,
        final_update=dict(RMS=rms,relative_L2=relative,max_tensor_relative_L2=relative_max),
        RNG=dict(all_four_saved_at_each_milestone=True,all100_Python_case_crop_replay_exact=True,all100_MANIQA_NumPy20_crop_draw_replay_exact=True,TorchCPU_exact_unchanged=True,CUDA_saved_seed_count=1,CUDA_actual_offsets=offsets,CUDA_offset_increment_per_update=increment,CUDA_replay_not_executed=True),
        median_step_seconds=statistics.median(r['step_seconds'] for r in rows[1:]),peak_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in rows),file_records=files,auditor_source_sha256=sha(__file__),limitations='No GPU/model rerun; CUDA RNG consistency is saved Philox-counter continuity, not CUDA replay. Runtime TF32 in trainer protocol is configured constants, not an added process getter; no current quality/score conclusion.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',out=str(a.out),sha256=sha(a.out),updates=100,final_update=result['final_update'],CUDA_initialized=False)))

if __name__=='__main__':main()
