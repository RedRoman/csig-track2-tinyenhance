"""CPU audit of actual structure-v2 PixG5i milestones and immutable logged prefix."""
from pathlib import Path
import argparse,hashlib,json,math,os,random
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
torch.set_num_threads(4)
R=Path('/cache/independent_sr_cuda_20261013');E=R/'experiments/pixg5i_hypir_distill_20261015'
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487';TEACHER='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c'
TRAINER='3c2e83ce965ce4dbc33fa5306679c8e969f1fe1477d41af27c9ed1fc6834225e'
PARENT='45c7f6bb414567f31b906fc996c037f31df7ce6aebb615c19183e02886950fab'
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
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',nargs='+',type=int,choices=[1,100,300,600,1200],default=[1]);a=p.parse_args()
steps=sorted(set([0]+a.steps));N=max(steps);out=E/'audits'/('structure_v2_training_steps'+'_'.join(map(str,a.steps))+'_cpu.json');need(not out.exists(),'Immutable new audit only')
need(not torch.cuda.is_initialized(),'CPU only');initialpath=E/'runs/pixg5i_hypir_medium/step600.pth';need(sha(initialpath)==PARENT,'Actual medium600 parent SHA')
initial=torch.load(initialpath,map_location='cpu',weights_only=True,mmap=True);results=[];cross=[]
for variant in ['structure_v2']:
    d=E/'runs'/('pixg5i_hypir_'+variant);cfg=read(d/'config.json');q=cfg['protocol'];pid=hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()
    need(pid==cfg['protocol_id'] and q['initial_checkpoint_sha256']==INITIAL and q['teacher_checkpoint_sha256']==TEACHER and TRAINER in q['source_sha256'].values(),'Sealed lineage/source')
    need(q['schema']=='pixg5i_HYPIR_structure_fidelity_v2' and q['state_tensor_count']==112 and q['parameter_elements']==3786764 and q['no_val_GT_training'] is True and q['objective']['NR_reward']==0. and q['objective']['original_output_anchor']==0.,'Original112/structure/noGT/noNR')
    need(q['phase_initial_checkpoint_sha256']==PARENT and q['parent_distillation_steps']==600 and q['parent']['raw']['sha256']==PARENT and q['parent']['optimizer_inherited'] is False and q['parent']['RNG_inherited'] is False,'Parent600/freshAdam/RNG reset provenance')
    need(q['objective']==dict(L1_teacher=1.,LPIPS_teacher=.05,gradient_teacher=.3,multiscale_L1_teacher=.5,multiscale_gradient_teacher=.3,multiscale_factors=[2,4],NR_reward=0.,original_output_anchor=0.) and q['seed']==20261016 and q['stage_steps']==1200 and q['lr_schedule']==dict(kind='cosine',initial=2e-5,final=2e-6,total_steps=1200),'Exact structure objective/LR/seed')
    for path,h in q['source_sha256'].items():need(sha(path)==h,'Frozen actual source changed')
    need(sha(q['teacher_manifest'])==q['teacher_manifest_sha256']=='f1a2004d3e4fa56272bc00183a84960510dcc6066bb98b0d29930d04a7642fd6','Frozen teacher manifest')
    inputs=read(d/'input_catalog.json');targets=read(d/'teacher_catalog.json');need(inputs==q['input_catalog'] and targets==q['teacher_catalog'] and [x['case'] for x in inputs]==[x['case'] for x in targets]==list(range(1,101)),'Test100 ordered catalogs')
    lines=(d/'train.jsonl').read_text().splitlines();need(len(lines)>=N,'Actual log prefix absent');rows=[json.loads(x) for x in lines[:N]]
    for i,row in enumerate(rows,1):
        need(finite(row) and row['local_step']==row['optimizer_step']==i and row['finite_parameter_gradients']==112 and 0<row['nonzero_parameter_gradients']<=112 and 0<row['actual_updated_tensors']<=112,'Actual finite112/update/counter log')
        expected=row['L1_teacher_raw']+.05*row['LPIPS_teacher_clamped']+.3*row['gradient_teacher_raw']+.5*row['multiscale_L1_teacher_raw']+.3*row['multiscale_gradient_teacher_raw']
        need(row['global_distillation_step']==600+i,'Total distillation step accounting')
        need(abs(expected-row['loss'])<5e-7 and abs(row['lr']-(2e-6+9e-6*(1+math.cos(math.pi*(i-1)/1199))))<1e-15 and row['grad_scaler_scale']==1024.,'Independent fidelity/LR/scaler recompute')
        need(row['teacher_png_sha256']==targets[row['case']-1]['sha256'],'Logged actual teacher catalog SHA')
    milestones={};states={}
    for step in steps:
        marker=d/f'checkpoint_step{step}.json';m=read(marker);raw=Path(m['raw_checkpoint']);resume=Path(m['resume_checkpoint'])
        need(raw==d/f'step{step}.pth' and resume==d/f'resume_step{step}.pt' and sha(raw)==m['raw_sha256'] and sha(resume)==m['resume_sha256'],'Actual whole file SHA/path')
        ck=torch.load(raw,map_location='cpu',weights_only=True,mmap=True);rs=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
        need(m['global_step']==rs['total_distillation_steps']==ck['origin']['total_distillation_steps']==600+step and m['phase_initial_checkpoint_sha256']==rs['phase_initial_checkpoint_sha256']==ck['origin']['parent_checkpoint_sha256']==PARENT and ck['origin']['optimizer_reset'] is True,'Actual parent/phase/global origin')
        need(ck['cfg']==rs['cfg']==initial['cfg'] and ck['reference_step']==30000 and ck['step']==rs['local_step']==rs['optimizer_step']==m['local_step']==m['optimizer_step']==step and ck['protocol_id']==rs['protocol_id']==m['protocol_id']==pid,'Exact cfg/steps/protocol')
        need(list(ck['state'])==list(rs['state'])==list(initial['state']) and len(ck['state'])==112 and sum(v.numel() for v in ck['state'].values())==3786764,'112 original state shape')
        for k,v in ck['state'].items():need(v.dtype==rs['state'][k].dtype==torch.float32 and v.shape==rs['state'][k].shape==initial['state'][k].shape and bool(torch.isfinite(v).all()) and bool(torch.isfinite(rs['state'][k]).all()) and torch.equal(v,rs['state'][k]),'Actual112 FP32 raw/resume equal')
        changed=sum(not torch.equal(v,initial['state'][k]) for k,v in ck['state'].items());need(changed==m['changed_FP32_state_tensors'] and (changed>0 if step else changed==0),'Exact parent600 phase0/real new update')
        opt=rs['optimizer'];need(len(opt['param_groups'])==1,'One Adam group');group=opt['param_groups'][0];ids=group['params']
        need(len(ids)==len(set(ids))==112 and len(opt['state'])==(112 if step else 0),'Fresh Adam112 state count')
        need(group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0 and group['foreach'] is False,'Fresh Adam parameters')
        if step:
            need(set(ids)==set(opt['state']) and abs(group['lr']-rows[step-1]['lr'])<1e-15,'Adam ids/LR')
            for ident,v in zip(ids,ck['state'].values()):
                st=opt['state'][ident];need(int(st['step'])==step and all(st[k].dtype==torch.float32 and st[k].shape==v.shape and bool(torch.isfinite(st[k]).all()) for k in ['exp_avg','exp_avg_sq']) and bool((st['exp_avg_sq']>=0).all()),'Actual FP32 moments/steps/shapes')
        rng={k:rs[k] for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']};random.Random().setstate(rng['python_rng']);np.random.RandomState().set_state(rng['numpy_rng']);torch.Generator(device='cpu').set_state(rng['torch_rng'])
        for k in ['torch_rng','cuda_rng']:need(rng[k].device.type=='cpu' and rng[k].dtype==torch.uint8 and rng[k].ndim==1 and rng[k].numel()>=16,'Saved RNG valid byte state')
        need(finite(rs['grad_scaler']) and rs['grad_scaler']['scale']==1024. and rs['grad_scaler']['_growth_tracker']==step and m['all4_RNG_saved'] and m['grad_scaler_saved'],'Actual scaler/RNG saved')
        states[step]=rng;milestones[str(step)]=dict(local_step=step,optimizer_step=step,raw_sha256=m['raw_sha256'],resume_sha256=m['resume_sha256'],marker_sha256=sha(marker),state_tensors=112,parameter_elements=3786764,raw_resume_FP32_exact=True,changed_FP32_tensors=changed,Adam_state_entries=len(opt['state']),Adam_all_finite_FP32=True,Adam_steps=[step] if step else [],four_RNG_valid=True,grad_scaler=rs['grad_scaler'])
        del ck,rs
    replay=random.Random();replay.setstate(states[0]['python_rng'])
    for i,row in enumerate(rows,1):
        case=replay.randint(1,100);w,h=inputs[case-1]['size'];xy=[8*replay.randrange((w-1024)//8+1),8*replay.randrange((h-1024)//8+1)]
        need(case==row['case'] and xy==row['context_crop_xy'] and [xy[0]+256,xy[1]+256]==row['loss_crop_xy'],'Actual all-prefix Python case/crop replay')
        if i in states:need(replay.getstate()==states[i]['python_rng'],'Exact milestone Python RNG')
    pre=read(d/'gradient_preflight_step1.json');need(pre['status']=='PASS_FINITE112_AND_REAL_UPDATE' and pre['row']==rows[0] and pre['FP32_master_all'] and pre['frozen_LPIPS'] and pre['teacher_detached'] and pre['phase_initial_checkpoint_sha256']==PARENT and pre['NR_reward']==0.,'Actual step1 gradient proof')
    component=pre['weighted_RGB_gradient_norms'];need(set(component)=={'L1','LPIPS','gradient','MS_L1','MS_gradient'} and all(math.isfinite(v) and v>0 for v in component.values()) and math.isfinite(pre['true_RGB_gradient_norm']) and pre['true_RGB_gradient_norm']>0,'All five actual weighted RGB gradients finite/nonzero')
    relation=read(d/'initial_deployment_relation.json');need(finite(relation),'Actual initial parent fold/AMP relation finite')
    endpoints={str(k):rdigest(v) for k,v in states.items()};cross.append(dict(data=[(r['case'],r['context_crop_xy'],r['teacher_png_sha256']) for r in rows],rng=endpoints))
    results.append(dict(run=str(d),variant=variant,protocol_id=pid,initial_package_sha256=INITIAL,teacher_raw_sha256=TEACHER,config_sha256=sha(d/'config.json'),milestones=milestones,logged_prefix_steps=N,log_prefix_sha256=hashlib.sha256(('\n'.join(lines[:N])+'\n').encode()).hexdigest(),all_prefix_losses_gradients_updates_finite=True,all_prefix_cases_crops_replayed=True,initial_deployment_relation=relation,weighted_RGB_gradient_norms=component,parent_checkpoint_sha256=PARENT,RNG_endpoint_digests=endpoints,teacher_PNG_bytes_rehashed_here=False))
need(not torch.cuda.is_initialized(),'No CUDA initialization')
report=dict(status='PASS_PIXG5I_STRUCTURE_FP32_ADAM_SCALER_RNG',auditor_source_sha256=sha(__file__),initial_package_sha256=INITIAL,teacher_raw_sha256=TEACHER,steps=steps,models=results,single_run_all_case_crop_replay_and_valid_RNG_endpoints=True,parent_medium600_checkpoint_sha256=PARENT,CPU_only=True,CUDA_initialized=False,quality_conclusion=False,mutable_status_not_locked=True,mutable_logs_checked_as_immutable_prefix_only=True,teacher_byte_scope='Frozen trainer verified all100 actual PNG SHA at initialization; this audit checks catalog/manifest/log identity, not full PNG reread')
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(status=report['status'],out=str(out),sha256=sha(out),steps=steps,models=[dict(variant=x['variant'],milestones=x['milestones']) for x in results])))
