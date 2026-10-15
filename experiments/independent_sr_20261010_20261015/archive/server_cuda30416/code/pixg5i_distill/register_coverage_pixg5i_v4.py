"""CPU-only actual PixG5i coverage-v4 milestone registration; default dry-run.

Requires an immutable real marker/raw/resume. Original inference is copied
unchanged; no CUDA, training, scoring, teacher inference or preference writes.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,shutil,sys
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013');E=R/'experiments/pixg5i_hypir_distill_20261015'
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'
TEACHER='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c'
TM='f1a2004d3e4fa56272bc00183a84960510dcc6066bb98b0d29930d04a7642fd6'
TRAIN='25624067758d5717e110ac7025c4e9eb8ee2ad8b2cfc292273581d6b5510fa08'
CLI='6cd1004d143f37879dfbe512bb8e1fc2e5018cb5371bf7d0b30a0a7870c9ee9d'
PARENT='24a68afe7c24c46fb60b768feea75ab5b32b2b681dc2a4be88765a0a05c54bfa'
PARENT_CM='398a259dcb8bd6616b6d3043ed414eba73ef63e400bb21eceaf19f1160081a56'
HEAD_NAMES=['outc.weight','outc.bias']
ADAPTER='70c5813d20597e7260ea51b60ccdc98291e77f18e56c1131a17c4022ae4c143a'
FILES={'runner.py':'6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e',
       'pix_model.py':'2f1cd3cd8311381e432138f489b01b982e08ebd5b09ecbc05ac87f7bfb577923',
       'config.yaml':'e13bab32185ac95f07ac36133eff89133a33ed90cc5ca534b7d8496bc457250d'}
def need(x,m):
    if not x:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def copy(src,dst):
    with Path(src).open('rb') as a,Path(dst).open('xb') as b:shutil.copyfileobj(a,b)
    need(sha(src)==sha(dst),'Copied deployment bytes differ')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step',type=int,choices=[200,600,1200,2000],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    run=E/'runs/pixg5i_hypir_coverage_v4';N=a.step;raw=run/f'step{N}.pth';resume=run/f'resume_step{N}.pt'
    marker_path=run/f'checkpoint_step{N}.json';m=read(marker_path);config=read(run/'config.json');q=config['protocol']
    need(m['protocol_id']==config['protocol_id']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest(),'Actual canonical training protocol differs')
    need(Path(m['raw_checkpoint']).resolve()==raw and Path(m['resume_checkpoint']).resolve()==resume and
         sha(raw)==m['raw_sha256'] and sha(resume)==m['resume_sha256'] and
         m['local_step']==m['optimizer_step']==N and m['global_step']==1800+N and m['reference_step']==30000 and
         m['parent_distillation_steps']==1800 and m['phase_initial_checkpoint_sha256']==PARENT and
         m['state_tensor_count']==112 and m['parameter_elements']==3786764 and m['no_val_GT_training'] is True and
         m['initial_package_checkpoint_sha256']==INITIAL and m['teacher_checkpoint_sha256']==TEACHER and m['NR_reward']==0,
         'Actual immutable checkpoint/lineage/counters differ')
    need(q['schema']=='pixg5i_HYPIR_coverage_fidelity_v4' and q['run']==run.name and
         q['initial_checkpoint_sha256']==INITIAL and q['teacher_checkpoint_sha256']==TEACHER and
         q['teacher_manifest_sha256']==TM and sha(q['teacher_manifest'])==TM and q['reference_step']==30000 and
         q['stage_steps']==2000 and q['state_tensor_count']==112 and q['parameter_elements']==3786764 and
         q['no_val_GT_training'] is True and q['used_as_G4_reference'] is False and
         q['context']==1024 and q['loss_center']==512 and q['halo']==256 and q['crop_grid']==8 and
         q['objective']==dict(L1_teacher=1.,LPIPS_teacher=.005,gradient_teacher=.3,multiscale_L1_teacher=.5,multiscale_gradient_teacher=.3,multiscale_factors=[2,4],NR_reward=0.,original_output_anchor=0.) and
         q['seed']==20261017 and q['lr_schedule']==dict(kind='two_group_warmup_hold_cosine',warmup_steps=50,warmup_initial_fraction=.1,hold_through_step=1000,body_maximum=1e-4,body_final=2e-5,head_maximum=2e-4,head_final=4e-5,total_steps=2000) and
         q['reset_head'] is False and m['reset_head'] is False and
         q['parameter_groups']==dict(body_tensors=110,head_names=HEAD_NAMES,head_elements=3468) and
         m['head_group_names']==HEAD_NAMES and m['head_group_elements']==3468 and
         q['parent_distillation_steps']==1800 and q['phase_initial_checkpoint_sha256']==PARENT and m['parent']==q['parent'],
         'Reviewed original-architecture test100 teacher-fidelity protocol differs')
    need(m['source_sha256']==q['source_sha256'] and TRAIN in q['source_sha256'].values() and CLI in q['source_sha256'].values(),'Actual frozen trainer/CLI differs')
    for path,h in q['source_sha256'].items():need(sha(path)==h,'Actual training source changed')
    source=[Path(x) for x,h in q['source_sha256'].items() if h==TRAIN];need(len(source)==1,'Unique actual trainer source required')
    need(Path(m['input_catalog'])==run/'input_catalog.json' and Path(m['teacher_catalog'])==run/'teacher_catalog.json' and
         sha(m['input_catalog'])==m['input_catalog_sha256'] and sha(m['teacher_catalog'])==m['teacher_catalog_sha256'],'Actual catalogs differ')
    inputs=read(m['input_catalog']);targets=read(m['teacher_catalog'])
    need(inputs==q['input_catalog'] and targets==q['teacher_catalog'] and
         [x['case'] for x in inputs]==[x['case'] for x in targets]==list(range(1,101)),'Test100 catalogs/order differ')
    mf=read(q['teacher_manifest']);tasks=[x for x in mf['protocol']['tasks'] if x['split']=='test'];records=[x for x in mf['records'] if x['split']=='test']
    for i,t,s,r in zip(inputs,targets,tasks,records):
        need(i['case']==t['case']==s['case']==r['case'] and i['sha256']==s['input_sha256']==t['input_sha256'] and
             Path(i['path']).resolve()==Path(s['input']).resolve() and t['sha256']==r['sha256'] and
             Path(t['path']).resolve()==(Path(q['teacher_manifest']).parent/r['output']).resolve() and i['size']==t['size']==s['size']==r['size'],
             'Actual frozen teacher/test100 catalog binding differs')
    plan_path=run/'training_plan.json'
    need(Path(m['training_plan']).resolve()==plan_path and Path(q['sampler']['plan']).resolve()==plan_path and
         sha(plan_path)==m['training_plan_sha256']==q['sampler']['plan_sha256'] and
         m['sampler_next_index']==N and q['sampler']['seed']==20261017 and q['sampler']['origin_grid']==8 and
         q['sampler']['selection_uses_input_dimensions_only'] is True,'Actual sealed sampler plan/index differs')
    plan=read(plan_path)
    spec=importlib.util.spec_from_file_location('_verified_coverage_training_plan_producer',source[0])
    producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
    need(plan==producer.build_training_plan(inputs,20261017) and len(plan)==2000,'Actual deterministic stratified plan replay differs')
    for cycle in range(20):
        need(sorted(x['case'] for x in plan[cycle*100:(cycle+1)*100])==list(range(1,101)),'Case100 without-replacement cycle differs')
    for case in range(1,101):
        need(sorted(tuple(x['stratum_xy']) for x in plan if x['case']==case)==sorted((x,y) for y in range(4) for x in range(5)),'Case twenty context strata differ')
    parent_cm_path=R/'submissions/pixg5i_hypir_headreset_v3_s600/candidate_manifest.json'
    need(sha(parent_cm_path)==PARENT_CM,'Actual registered headreset-v3 local600/global1800 parent changed')
    parent_cm=read(parent_cm_path);parent=q['parent']
    need(sha(parent['candidate_manifest']['path'])==parent['candidate_manifest']['sha256']==PARENT_CM and
         Path(parent['candidate_manifest']['path']).resolve()==parent_cm_path,'Actual parent candidate-manifest reference differs')
    need(parent_cm['checkpoint_sha256']==PARENT and parent_cm['local_step']==parent_cm['optimizer_step']==600 and parent_cm['global_step']==parent['parent_global_distillation_step']==1800 and
         parent['parent_local_step']==parent['parent_optimizer_step']==600 and parent['optimizer_inherited'] is False and
         parent['RNG_inherited'] is False and parent['raw_resume_FP32_exact'] is True,
         'Honest headreset-v3 local600/global1800 initialization/fresh-phase metadata differs')
    need(q['parent_historical_head_reset']==parent_cm['reset_head'],'Historical reset metadata must be separate from this no-reset phase')
    for key in ['marker','config','raw','resume']:
        need(sha(parent[key]['path'])==parent[key]['sha256'],'Actual immutable phase parent file changed: '+key)
    need(parent['raw']['sha256']==PARENT and Path(parent['raw']['path']).resolve()==Path(parent_cm['checkpoint']).resolve() and
         all(parent['marker'][k]==parent_cm['training_provenance']['marker'][k] for k in ['path','sha256']) and
         all(parent['config'][k]==parent_cm['training_provenance']['config'][k] for k in ['path','sha256']) and
         all(parent['resume'][k]==parent_cm['training_provenance']['resume'][k] for k in ['path','sha256']),
         'Phase parent differs from actual sealed headreset-v3 local600/global1800 registration')
    import torch
    torch.set_num_threads(4)
    initial_path=R/'source_payload/pixg5i_original/model_dir/model.pth';need(sha(initial_path)==INITIAL,'Original PixG5i weights changed')
    ck=torch.load(raw,map_location='cpu',weights_only=False,mmap=True);rs=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(initial_path,map_location='cpu',weights_only=False,mmap=True)
    phase_initial=torch.load(parent['raw']['path'],map_location='cpu',weights_only=False,mmap=True)
    need(phase_initial['cfg']==initial['cfg'] and phase_initial['step']==600 and phase_initial['reference_step']==30000 and
         list(phase_initial['state'])==list(initial['state']) and rs['phase_initial_checkpoint_sha256']==PARENT and
         rs['parent_distillation_steps']==1800 and rs['total_distillation_steps']==1800+N,
         'Actual phase initial checkpoint/resume lineage differs')
    need(list(phase_initial['state'])[-2:]==HEAD_NAMES and
         tuple(phase_initial['state']['outc.weight'].shape)==(12,32,3,3) and
         tuple(phase_initial['state']['outc.bias'].shape)==(12,) and
         sum(phase_initial['state'][k].numel() for k in HEAD_NAMES)==3468,'Actual original residual head shape differs')
    phase_initial_state=phase_initial['state'] # All112 inherited exactly; no new head reset.
    h=hashlib.sha256()
    for k,v in phase_initial_state.items():
        h.update(json.dumps([k,list(v.shape),str(v.dtype)],separators=(',',':')).encode());h.update(b'\0')
        h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    need(h.hexdigest()==q['initial_state_fingerprint_sha256'],'Actual all112-parent initial fingerprint differs')
    need(rs['sampler_plan_sha256']==m['training_plan_sha256'] and rs['sampler_next_index']==N,'Actual resume sampler next-index/plan differs')
    need(ck['cfg']==rs['cfg']==initial['cfg'] and ck['step']==rs['local_step']==rs['optimizer_step']==N and ck['reference_step']==30000 and
         ck['protocol_id']==rs['protocol_id']==m['protocol_id'] and list(ck['state'])==list(rs['state'])==list(initial['state']) and
         len(ck['state'])==112 and sum(v.numel() for v in ck['state'].values())==3786764,'Actual raw/resume/original architecture differs')
    for k,v in ck['state'].items():
        need(v.dtype==rs['state'][k].dtype==torch.float32 and v.shape==rs['state'][k].shape==initial['state'][k].shape==phase_initial['state'][k].shape and
             phase_initial['state'][k].dtype==torch.float32 and bool(torch.isfinite(phase_initial['state'][k]).all()) and
             bool(torch.isfinite(v).all()) and bool(torch.isfinite(rs['state'][k]).all()) and torch.equal(v,rs['state'][k]),'Actual FP32 raw/resume tensors differ')
    changed=sum(not torch.equal(v,phase_initial_state[k]) for k,v in ck['state'].items())
    body_changed=sum(not torch.equal(v,phase_initial['state'][k]) for k,v in ck['state'].items() if k not in HEAD_NAMES)
    need(body_changed==m['body_changed_FP32_tensors_vs_parent'],'Actual body110 change count differs')
    need(changed==m['changed_FP32_state_tensors'] and changed>0 and ck['origin']==dict(model_family='PixG5i_PixUNet2',
         initial_package_checkpoint_sha256=INITIAL,teacher_checkpoint_sha256=TEACHER,new_training_steps=N,parent_distillation_steps=1800,
         total_distillation_steps=1800+N,parent_checkpoint_sha256=PARENT,optimizer_reset=True,reset_head=False,head_group_names=HEAD_NAMES,head_group_elements=3468,NR_reward=0.),'Actual phase-changed tensors/origin differ')
    opt=rs['optimizer'];groups=opt['param_groups']
    need(len(groups)==2 and [g['group_name'] for g in groups]==['body','head'] and
         [len(g['params']) for g in groups]==[110,2],'Exact two-group body110/head2 Adam required')
    ids=[i for g in groups for i in g['params']]
    need(len(ids)==len(set(ids))==len(opt['state'])==m['optimizer_state_entries']==112 and set(ids)==set(opt['state']),
         'Actual two-group Adam IDs/counts differ')
    def scheduled_lr(step,maximum,final):
        if step<=50:return maximum*(.1+.9*(step-1)/49)
        if step<=1000:return maximum
        return final+.5*(maximum-final)*(1+math.cos(math.pi*(step-1000)/1000))
    for group,lr0,lr1 in zip(groups,[1e-4,2e-4],[2e-5,4e-5]):
        need(group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0 and group['foreach'] is False and
             abs(group['lr']-scheduled_lr(N,lr0,lr1))<1e-15,
             'Actual body/head Adam hyperparameters/LR differ')
    for ident,v in zip(ids,ck['state'].values()):
        st=opt['state'][ident];need(int(st['step'])==N and all(st[k].dtype==torch.float32 and st[k].shape==v.shape and bool(torch.isfinite(st[k]).all()) for k in ['exp_avg','exp_avg_sq']),'Actual finite Adam moments/counters differ')
    need({'python_rng','numpy_rng','torch_rng','cuda_rng','grad_scaler'}<=set(rs) and m['all4_RNG_saved'] is True and
         m['grad_scaler_saved'] is True and m['all_FP32_finite'] is True and not torch.cuda.is_initialized(),'CPU registration/RNG/scaler contract differs')
    src=R/'source_payload/pixg5i_original/model_dir';adapter=Path(__file__).with_name('pixg5i_protocol_adapter.py')
    need(sha(adapter)==ADAPTER and all(sha(src/k)==h for k,h in FILES.items()),'Original deployment sources changed')
    label=f'pixg5i_hypir_coverage_v4_s{N}';md=R/'submissions'/label/'model_dir';need(not md.parent.exists(),'Never overwrite a registered model')
    provenance=dict(config=ref(run/'config.json'),marker=ref(marker_path),input_catalog=ref(m['input_catalog']),teacher_catalog=ref(m['teacher_catalog']),resume=ref(resume),training_source=ref(source[0]),training_plan=ref(plan_path))
    teacher=dict(raw_sha256=TEACHER,outputs=str(Path(q['teacher_manifest']).parent/'render/test'),applied_to_this_checkpoint=True,render_manifest=ref(q['teacher_manifest']))
    binding=dict(schema='pixg5i_hypir_distill_binding_v1',label=label,stage='distilled_checkpoint',checkpoint_sha256=m['raw_sha256'],
        initial_package_checkpoint_sha256=INITIAL,archive_sha256='f8e154da5ea9cb2fe165a316f824003459661945e6bbe22efd025d8592d573d6',
        training_provenance=provenance,teacher=teacher,used_as_G4_reference=False,variant='coverage_v4',local_step=N,optimizer_step=N,reference_step=30000,
        training_phase='coverage_v4',global_step=1800+N,parent_distillation_steps=1800,total_distillation_steps=1800+N,
        phase_initial_checkpoint_sha256=PARENT,phase_parent=parent,parent_candidate_manifest=ref(parent_cm_path),
        reset_head=False,parent_historical_head_reset=q['parent_historical_head_reset'],parameter_groups=q['parameter_groups'],
        sampler_plan_sha256=m['training_plan_sha256'],sampler_next_index=N,phase_initial_state_fingerprint_sha256=q['initial_state_fingerprint_sha256'])
    proof=dict(status='CPU_ACTUAL_TRAINED_CHECKPOINT_VALIDATED',model_dir=str(md),binding=binding,actual_changed_FP32_tensors_vs_all112_parent_initial=changed,actual_body_changed_FP32_tensors_vs_parent=body_changed,
        actual112_raw_resume_equal_FP32_finite=True,actual112_Adam_moments_finite_and_counters=True,all4_RNG_saved=True,
        teacher_catalog_bound_to_fixed_manifest=True,teacher_training_PNG_bytes_rehashed_here=False,
        actual_sampler_plan_regenerated_matches=True,sampler_next_index=N,training_logs_replayed_here=False,
        sampler_scope='Every100step cycle visits all100cases once; twenty5x4 context-origin strata percase over2000planned steps; not exhaustive pixels',CPU_only=True,CUDA_initialized=False)
    if a.execute:
        md.parent.mkdir(exist_ok=False);md.mkdir()
        for old,new in [('runner.py','reference_runner.py'),('pix_model.py','pix_model.py'),('config.yaml','config.yaml')]:copy(src/old,md/new)
        copy(raw,md/'model.pth');copy(adapter,md/'runner.py');write(md/'pixg5i_binding.json',binding)
        cm=dict(schema='pixg5i_registered_candidate_v1',label=label,model_dir=str(md),stage='distilled_checkpoint',variant='coverage_v4',
            model_type='supplied_pixG5i_pixel_UNet',checkpoint=str(raw),checkpoint_sha256=m['raw_sha256'],deployed_checkpoint=str(md/'model.pth'),
            initial_package_checkpoint_sha256=INITIAL,runner_sha256=ADAPTER,binding=ref(md/'pixg5i_binding.json'),files={x.name:ref(x) for x in md.iterdir() if x.is_file()},
            training_provenance=provenance,teacher=teacher,local_step=N,optimizer_step=N,reference_step=30000,
            training_phase='coverage_v4',global_step=1800+N,parent_distillation_steps=1800,total_distillation_steps=1800+N,
            phase_initial_checkpoint_sha256=PARENT,phase_parent=parent,parent_candidate_manifest=ref(parent_cm_path),
            reset_head=False,parent_historical_head_reset=q['parent_historical_head_reset'],parameter_groups=q['parameter_groups'],
            sampler_plan_sha256=m['training_plan_sha256'],sampler_next_index=N,phase_initial_state_fingerprint_sha256=q['initial_state_fingerprint_sha256'],
            registration_CPU_proof=proof,quality_reuse_claim=False,used_as_G4_reference=False,registration_source=ref(__file__),CPU_only=True,GPU_used=False,
            original_inference_source_unchanged=True)
        write(md.parent/'candidate_manifest.json',cm)
        proof=dict(status='REGISTERED_REAL_TRAINED_PIXG5I_CPU_ONLY',model_dir=str(md),label=label,candidate_manifest=ref(md.parent/'candidate_manifest.json'),checkpoint_sha256=m['raw_sha256'],GPU_used=False)
    print(json.dumps(proof,ensure_ascii=False))
if __name__=='__main__':main()
