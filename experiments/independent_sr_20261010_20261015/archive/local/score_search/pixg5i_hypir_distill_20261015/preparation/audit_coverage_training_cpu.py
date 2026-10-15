"""CPU audit of actual coverage-v4 PixG5i milestones and immutable logged prefix."""
from pathlib import Path
import argparse,hashlib,json,math,os,random,sys
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import torch,numpy as np
torch.set_num_threads(4)
R=Path('/cache/independent_sr_cuda_20261013');E=R/'experiments/pixg5i_hypir_distill_20261015'
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487';TEACHER='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c'
TRAINER='25624067758d5717e110ac7025c4e9eb8ee2ad8b2cfc292273581d6b5510fa08'
PARENT='24a68afe7c24c46fb60b768feea75ab5b32b2b681dc2a4be88765a0a05c54bfa'
def need(x,m):
    if not x:raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):
    if torch.is_tensor(x):return dict(dtype=str(x.dtype),shape=list(x.shape),SHA=hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())
    if isinstance(x,np.ndarray):return dict(dtype=str(x.dtype),shape=list(x.shape),SHA=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
    if isinstance(x,(tuple,list)):return [digest(v) for v in x]
    if isinstance(x,dict):return {k:digest(v) for k,v in x.items()}
    return x
def rdigest(x):return hashlib.sha256(json.dumps(digest(x),sort_keys=True).encode()).hexdigest()
def finite(x):
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,list):return all(finite(v) for v in x)
    return not isinstance(x,(int,float)) or math.isfinite(x)
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',nargs='+',type=int,choices=[1,200,600,1200,2000],default=[1]);a=p.parse_args()
steps=sorted(set([0]+a.steps));N=max(steps);out=E/'audits'/('coverage_v4_training_steps'+'_'.join(map(str,a.steps))+'_cpu.json');need(not out.exists(),'Immutable new audit only')
need(not torch.cuda.is_initialized(),'CPU only');initialpath=E/'runs/pixg5i_hypir_headreset_v3/step600.pth';need(sha(initialpath)==PARENT,'Actual medium600 parent SHA')
parent_original=torch.load(initialpath,map_location='cpu',weights_only=True,mmap=True);initial=dict(cfg=parent_original['cfg'],state={k:v.clone() for k,v in parent_original['state'].items()});head_names=['outc.weight','outc.bias'];need(list(initial['state'])[-2:]==head_names and sum(initial['state'][k].numel() for k in head_names)==3468,'Original head2/3468')
results=[];cross=[]
for variant in ['coverage_v4']:
    d=E/'runs'/('pixg5i_hypir_'+variant);cfg=read(d/'config.json');q=cfg['protocol'];pid=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
    need(pid==cfg['protocol_id'] and q['initial_checkpoint_sha256']==INITIAL and q['teacher_checkpoint_sha256']==TEACHER and TRAINER in q['source_sha256'].values(),'Sealed lineage/source')
    need(q['schema']=='pixg5i_HYPIR_coverage_fidelity_v4' and q['state_tensor_count']==112 and q['parameter_elements']==3786764 and q['no_val_GT_training'] is True and q['objective']['NR_reward']==0. and q['objective']['original_output_anchor']==0.,'Original112/structure/noGT/noNR')
    need(q['phase_initial_checkpoint_sha256']==PARENT and q['parent_distillation_steps']==1800 and q['parent']['raw']['sha256']==PARENT and q['parent']['optimizer_inherited'] is False and q['parent']['RNG_inherited'] is False,'Parent600/freshAdam/RNG reset provenance')
    need(q['objective']==dict(L1_teacher=1.,LPIPS_teacher=.005,gradient_teacher=.3,multiscale_L1_teacher=.5,multiscale_gradient_teacher=.3,multiscale_factors=[2,4],NR_reward=0.,original_output_anchor=0.) and q['seed']==20261017 and q['reset_head'] is False and q['stage_steps']==2000 and q['lr_schedule']==dict(kind='two_group_warmup_hold_cosine',warmup_steps=50,warmup_initial_fraction=.1,hold_through_step=1000,body_maximum=1e-4,body_final=2e-5,head_maximum=2e-4,head_final=4e-5,total_steps=2000),'Exact structure objective/LR/seed')
    for path,h in q['source_sha256'].items():need(sha(path)==h,'Frozen actual source changed')
    need(sha(q['teacher_manifest'])==q['teacher_manifest_sha256']=='f1a2004d3e4fa56272bc00183a84960510dcc6066bb98b0d29930d04a7642fd6','Frozen teacher manifest')
    inputs=read(d/'input_catalog.json');targets=read(d/'teacher_catalog.json');need(inputs==q['input_catalog'] and targets==q['teacher_catalog'] and [x['case'] for x in inputs]==[x['case'] for x in targets]==list(range(1,101)),'Test100 ordered catalogs')
    planpath=d/'training_plan.json';plan=read(planpath);plan_sha=sha(planpath);need(plan_sha==q['sampler']['plan_sha256']=='5466f898c854b8337d16879a867b04b14b631959df18c6baca817c9eaafe0d28' and len(plan)==2000,'Actual seeded plan SHA/length')
    import importlib.util
    trainer_path=next(Path(k) for k,h in q['source_sha256'].items() if h==TRAINER);spec=importlib.util.spec_from_file_location('_audited_coverage_source',trainer_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    need(module.build_training_plan(inputs,20261017)==plan,'Exact actual100catalog CPU sampler replay')
    for start in range(0,2000,100):need(sorted(v['case'] for v in plan[start:start+100])==list(range(1,101)),'Every epoch all100 exactly once')
    for case in range(1,101):need({tuple(v['stratum_xy']) for v in plan if v['case']==case}=={(x,y) for y in range(4) for x in range(5)},'Each case all20 strata')
    def lr_expected(step,maximum,final):
        if step<=50:return maximum*(.1+.9*(step-1)/49)
        if step<=1000:return maximum
        return final+.5*(maximum-final)*(1+math.cos(math.pi*(step-1000)/1000))
    lines=(d/'train.jsonl').read_text().splitlines();need(len(lines)>=N,'Actual log prefix absent');rows=[json.loads(x) for x in lines[:N]]
    for i,row in enumerate(rows,1):
        need(finite(row) and row['local_step']==row['optimizer_step']==i and row['finite_parameter_gradients']==112 and 0<row['nonzero_parameter_gradients']<=112 and 0<row['actual_updated_tensors']<=112,'Actual finite112/update/counter log')
        expected=row['L1_teacher_raw']+.005*row['LPIPS_teacher_clamped']+.3*row['gradient_teacher_raw']+.5*row['multiscale_L1_teacher_raw']+.3*row['multiscale_gradient_teacher_raw']
        need(row['global_distillation_step']==1800+i,'Total distillation step accounting')
        need(abs(expected-row['loss'])<5e-7 and abs(row['lr']-lr_expected(i,1e-4,2e-5))<1e-15 and abs(row['head_lr']-lr_expected(i,2e-4,4e-5))<1e-15 and row['grad_scaler_scale']==1024.,'Independent fidelity/two-group LR/scaler recompute')
        need(row['actual_body_updated_tensors']+row['actual_head_updated_tensors']==row['actual_updated_tensors'],'Actual first body0/head2 update and later split counts')
        need(row['teacher_png_sha256']==targets[row['case']-1]['sha256'],'Logged actual teacher catalog SHA')
    milestones={};states={}
    for step in steps:
        marker=d/f'checkpoint_step{step}.json';m=read(marker);raw=Path(m['raw_checkpoint']);resume=Path(m['resume_checkpoint'])
        need(raw==d/f'step{step}.pth' and resume==d/f'resume_step{step}.pt' and sha(raw)==m['raw_sha256'] and sha(resume)==m['resume_sha256'],'Actual whole file SHA/path')
        ck=torch.load(raw,map_location='cpu',weights_only=True,mmap=True);rs=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
        need(m['global_step']==rs['total_distillation_steps']==ck['origin']['total_distillation_steps']==1800+step and m['phase_initial_checkpoint_sha256']==rs['phase_initial_checkpoint_sha256']==ck['origin']['parent_checkpoint_sha256']==PARENT and ck['origin']['optimizer_reset'] is True,'Actual parent/phase/global origin')
        need(ck['cfg']==rs['cfg']==initial['cfg'] and ck['reference_step']==30000 and ck['step']==rs['local_step']==rs['optimizer_step']==m['local_step']==m['optimizer_step']==step and ck['protocol_id']==rs['protocol_id']==m['protocol_id']==pid,'Exact cfg/steps/protocol')
        need(list(ck['state'])==list(rs['state'])==list(initial['state']) and len(ck['state'])==112 and sum(v.numel() for v in ck['state'].values())==3786764,'112 original state shape')
        for k,v in ck['state'].items():need(v.dtype==rs['state'][k].dtype==torch.float32 and v.shape==rs['state'][k].shape==initial['state'][k].shape and bool(torch.isfinite(v).all()) and bool(torch.isfinite(rs['state'][k]).all()) and torch.equal(v,rs['state'][k]),'Actual112 FP32 raw/resume equal')
        changed=sum(not torch.equal(v,initial['state'][k]) for k,v in ck['state'].items());need(changed==m['changed_FP32_state_tensors'] and (changed>0 if step else changed==0),'Exact parent600 phase0/real new update')
        body_changed=sum(not torch.equal(v,parent_original['state'][k]) for k,v in ck['state'].items() if k not in head_names);need(body_changed==m['body_changed_FP32_tensors_vs_parent'] and m['reset_head'] is False and m['head_group_names']==head_names and m['head_group_elements']==3468 and m['training_plan_sha256']==rs['sampler_plan_sha256']==plan_sha and m['sampler_next_index']==rs['sampler_next_index']==step,'Actual reset/body110 marker')
        if step==0:need(all(torch.equal(v,parent_original['state'][k]) for k,v in ck['state'].items()),'Initial all112 exact parent; no reset')
        opt=rs['optimizer'];groups=opt['param_groups'];need(len(groups)==2 and [g['group_name'] for g in groups]==['body','head'] and [len(g['params']) for g in groups]==[110,2],'Two exact body/head Adam groups');ids=[x for group in groups for x in group['params']]
        need(len(ids)==len(set(ids))==112 and len(opt['state'])==(112 if step else 0),'Fresh Adam112 state count')
        need(all(group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0 and group['foreach'] is False for group in groups),'Fresh two-group Adam parameters')
        if step:
            need(set(ids)==set(opt['state']) and abs(groups[0]['lr']-rows[step-1]['lr'])<1e-15 and abs(groups[1]['lr']-rows[step-1]['head_lr'])<1e-15,'Adam ids/two LR')
            for ident,v in zip(ids,ck['state'].values()):
                st=opt['state'][ident];need(int(st['step'])==step and all(st[k].dtype==torch.float32 and st[k].shape==v.shape and bool(torch.isfinite(st[k]).all()) for k in ['exp_avg','exp_avg_sq']) and bool((st['exp_avg_sq']>=0).all()),'Actual FP32 moments/steps/shapes')
        rng={k:rs[k] for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']};random.Random().setstate(rng['python_rng']);np.random.RandomState().set_state(rng['numpy_rng']);torch.Generator(device='cpu').set_state(rng['torch_rng'])
        for k in ['torch_rng','cuda_rng']:need(rng[k].device.type=='cpu' and rng[k].dtype==torch.uint8 and rng[k].ndim==1 and rng[k].numel()>=16,'Saved RNG valid byte state')
        need(finite(rs['grad_scaler']) and rs['grad_scaler']['scale']==1024. and rs['grad_scaler']['_growth_tracker']==step and m['all4_RNG_saved'] and m['grad_scaler_saved'],'Actual scaler/RNG saved')
        states[step]=rng;milestones[str(step)]=dict(local_step=step,optimizer_step=step,raw_sha256=m['raw_sha256'],resume_sha256=m['resume_sha256'],marker_sha256=sha(marker),state_tensors=112,parameter_elements=3786764,raw_resume_FP32_exact=True,changed_FP32_tensors=changed,body_changed_vs_parent=body_changed,reset_head=False,head_group_names=head_names,head_group_elements=3468,sampler_plan_sha256=plan_sha,sampler_next_index=step,Adam_groups=[dict(name=g['group_name'],parameters=len(g['params']),lr=g['lr']) for g in groups],Adam_state_entries=len(opt['state']),Adam_all_finite_FP32=True,Adam_steps=[step] if step else [],four_RNG_valid=True,grad_scaler=rs['grad_scaler'])
        del ck,rs
    for i,row in enumerate(rows,1):
        sample=plan[i-1];case=sample['case'];xy=sample['context_crop_xy'];w,h=inputs[case-1]['size']
        need(case==row['case'] and xy==row['context_crop_xy'] and [xy[0]+256,xy[1]+256]==row['loss_crop_xy'] and sample['cycle']==row['sampler_cycle'] and sample['stratum_xy']==row['sampler_stratum_xy'] and row['sampler_plan_sha256']==plan_sha,'Actual all-prefix sampler case/crop replay')
        need(xy[0]%8==xy[1]%8==0 and 0<=xy[0]<=w-1024 and 0<=xy[1]<=h-1024,'Actual native context in bounds')
    pre=read(d/'gradient_preflight_step1.json');need(pre['status']=='PASS_FINITE112_AND_REAL_UPDATE' and pre['row']==rows[0] and pre['FP32_master_all'] and pre['frozen_LPIPS'] and pre['teacher_detached'] and pre['phase_initial_checkpoint_sha256']==PARENT and pre['NR_reward']==0.,'Actual step1 gradient proof')
    component=pre['weighted_RGB_gradient_norms'];need(set(component)=={'L1','LPIPS','gradient','MS_L1','MS_gradient'} and all(math.isfinite(v) and v>0 for v in component.values()) and math.isfinite(pre['true_RGB_gradient_norm']) and pre['true_RGB_gradient_norm']>0,'All five actual weighted RGB gradients finite/nonzero')
    relation=read(d/'initial_deployment_relation.json');need(finite(relation),'Actual initial parent FP32/FP16 folded relation finite; no identity claim')
    endpoints={str(k):rdigest(v) for k,v in states.items()};cross.append(dict(data=[(r['case'],r['context_crop_xy'],r['teacher_png_sha256']) for r in rows],rng=endpoints))
    results.append(dict(run=str(d),variant=variant,protocol_id=pid,initial_package_sha256=INITIAL,teacher_raw_sha256=TEACHER,config_sha256=sha(d/'config.json'),milestones=milestones,logged_prefix_steps=N,log_prefix_sha256=hashlib.sha256(('\n'.join(lines[:N])+'\n').encode()).hexdigest(),all_prefix_losses_gradients_updates_finite=True,all_prefix_cases_crops_replayed=True,initial_deployment_relation=relation,weighted_RGB_gradient_norms=component,parent_checkpoint_sha256=PARENT,RNG_endpoint_digests=endpoints,teacher_PNG_bytes_rehashed_here=False))
need(not torch.cuda.is_initialized(),'No CUDA initialization')
report=dict(status='PASS_PIXG5I_COVERAGE_FP32_TWO_ADAM_SCALER_RNG',auditor_source_sha256=sha(__file__),initial_package_sha256=INITIAL,teacher_raw_sha256=TEACHER,steps=steps,models=results,single_run_all_case_crop_replay_and_valid_RNG_endpoints=True,parent_global1800_checkpoint_sha256=PARENT,CPU_only=True,CUDA_initialized=False,quality_conclusion=False,mutable_status_not_locked=True,mutable_logs_checked_as_immutable_prefix_only=True,teacher_byte_scope='Frozen trainer verified all100 actual PNG SHA at initialization; this audit checks catalog/manifest/log identity, not full PNG reread')
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(status=report['status'],out=str(out),sha256=sha(out),steps=steps,models=[dict(variant=x['variant'],milestones=x['milestones']) for x in results])))
