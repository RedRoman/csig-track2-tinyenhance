"""CPU-only actual PixG5i structure-v2 milestone registration; default dry-run.

Requires an immutable real marker/raw/resume. Original inference is copied
unchanged; no CUDA, training, scoring, teacher inference or preference writes.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,sys
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
R=Path('/cache/independent_sr_cuda_20261013');E=R/'experiments/pixg5i_hypir_distill_20261015'
INITIAL='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'
TEACHER='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c'
TM='f1a2004d3e4fa56272bc00183a84960510dcc6066bb98b0d29930d04a7642fd6'
TRAIN='3c2e83ce965ce4dbc33fa5306679c8e969f1fe1477d41af27c9ed1fc6834225e'
CLI='1e0704c62cf6e4dae8993311b49138ca398a866162799d262ba70a5a1852d589'
PARENT='45c7f6bb414567f31b906fc996c037f31df7ce6aebb615c19183e02886950fab'
PARENT_CM='1c5a7813c254ace9415486dd95fec8bf4412860f8ed7286e6989f1ab83b11b35'
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
    p.add_argument('--step',type=int,choices=[100,300,600,1200],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    run=E/'runs/pixg5i_hypir_structure_v2';N=a.step;raw=run/f'step{N}.pth';resume=run/f'resume_step{N}.pt'
    marker_path=run/f'checkpoint_step{N}.json';m=read(marker_path);config=read(run/'config.json');q=config['protocol']
    need(m['protocol_id']==config['protocol_id']==hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest(),'Actual canonical training protocol differs')
    need(Path(m['raw_checkpoint']).resolve()==raw and Path(m['resume_checkpoint']).resolve()==resume and
         sha(raw)==m['raw_sha256'] and sha(resume)==m['resume_sha256'] and
         m['local_step']==m['optimizer_step']==N and m['global_step']==600+N and m['reference_step']==30000 and
         m['parent_distillation_steps']==600 and m['phase_initial_checkpoint_sha256']==PARENT and
         m['state_tensor_count']==112 and m['parameter_elements']==3786764 and m['no_val_GT_training'] is True and
         m['initial_package_checkpoint_sha256']==INITIAL and m['teacher_checkpoint_sha256']==TEACHER and m['NR_reward']==0,
         'Actual immutable checkpoint/lineage/counters differ')
    need(q['schema']=='pixg5i_HYPIR_structure_fidelity_v2' and q['run']==run.name and
         q['initial_checkpoint_sha256']==INITIAL and q['teacher_checkpoint_sha256']==TEACHER and
         q['teacher_manifest_sha256']==TM and sha(q['teacher_manifest'])==TM and q['reference_step']==30000 and
         q['stage_steps']==1200 and q['state_tensor_count']==112 and q['parameter_elements']==3786764 and
         q['no_val_GT_training'] is True and q['used_as_G4_reference'] is False and
         q['context']==1024 and q['loss_center']==512 and q['halo']==256 and q['crop_grid']==8 and
         q['objective']==dict(L1_teacher=1.,LPIPS_teacher=.05,gradient_teacher=.3,multiscale_L1_teacher=.5,multiscale_gradient_teacher=.3,multiscale_factors=[2,4],NR_reward=0.,original_output_anchor=0.) and
         q['seed']==20261016 and q['lr_schedule']==dict(kind='cosine',initial=2e-5,final=2e-6,total_steps=1200) and
         q['parent_distillation_steps']==600 and q['phase_initial_checkpoint_sha256']==PARENT and m['parent']==q['parent'],
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
    parent_cm_path=R/'submissions/pixg5i_hypir_medium_s600/candidate_manifest.json'
    need(sha(parent_cm_path)==PARENT_CM,'Actual registered medium600 parent changed')
    parent_cm=read(parent_cm_path);parent=q['parent']
    need(parent_cm['checkpoint_sha256']==PARENT and parent_cm['local_step']==parent_cm['optimizer_step']==600 and
         parent['parent_local_step']==parent['parent_optimizer_step']==600 and parent['optimizer_inherited'] is False and
         parent['RNG_inherited'] is False and parent['raw_resume_FP32_exact'] is True,
         'Honest medium600 initialization/fresh-phase metadata differs')
    for key in ['marker','config','raw','resume']:
        need(sha(parent[key]['path'])==parent[key]['sha256'],'Actual immutable phase parent file changed: '+key)
    need(parent['raw']['sha256']==PARENT and Path(parent['raw']['path']).resolve()==Path(parent_cm['checkpoint']).resolve() and
         all(parent['marker'][k]==parent_cm['training_provenance']['marker'][k] for k in ['path','sha256']) and
         all(parent['config'][k]==parent_cm['training_provenance']['config'][k] for k in ['path','sha256']) and
         all(parent['resume'][k]==parent_cm['training_provenance']['resume'][k] for k in ['path','sha256']),
         'Phase parent differs from actual sealed medium600 registration')
    import torch
    torch.set_num_threads(4)
    initial_path=R/'source_payload/pixg5i_original/model_dir/model.pth';need(sha(initial_path)==INITIAL,'Original PixG5i weights changed')
    ck=torch.load(raw,map_location='cpu',weights_only=False,mmap=True);rs=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(initial_path,map_location='cpu',weights_only=False,mmap=True)
    phase_initial=torch.load(parent['raw']['path'],map_location='cpu',weights_only=False,mmap=True)
    need(phase_initial['cfg']==initial['cfg'] and phase_initial['step']==600 and phase_initial['reference_step']==30000 and
         list(phase_initial['state'])==list(initial['state']) and rs['phase_initial_checkpoint_sha256']==PARENT and
         rs['parent_distillation_steps']==600 and rs['total_distillation_steps']==600+N,
         'Actual phase initial checkpoint/resume lineage differs')
    need(ck['cfg']==rs['cfg']==initial['cfg'] and ck['step']==rs['local_step']==rs['optimizer_step']==N and ck['reference_step']==30000 and
         ck['protocol_id']==rs['protocol_id']==m['protocol_id'] and list(ck['state'])==list(rs['state'])==list(initial['state']) and
         len(ck['state'])==112 and sum(v.numel() for v in ck['state'].values())==3786764,'Actual raw/resume/original architecture differs')
    for k,v in ck['state'].items():
        need(v.dtype==rs['state'][k].dtype==torch.float32 and v.shape==rs['state'][k].shape==initial['state'][k].shape==phase_initial['state'][k].shape and
             phase_initial['state'][k].dtype==torch.float32 and bool(torch.isfinite(phase_initial['state'][k]).all()) and
             bool(torch.isfinite(v).all()) and bool(torch.isfinite(rs['state'][k]).all()) and torch.equal(v,rs['state'][k]),'Actual FP32 raw/resume tensors differ')
    changed=sum(not torch.equal(v,phase_initial['state'][k]) for k,v in ck['state'].items())
    need(changed==m['changed_FP32_state_tensors'] and changed>0 and ck['origin']==dict(model_family='PixG5i_PixUNet2',
         initial_package_checkpoint_sha256=INITIAL,teacher_checkpoint_sha256=TEACHER,new_training_steps=N,parent_distillation_steps=600,
         total_distillation_steps=600+N,parent_checkpoint_sha256=PARENT,optimizer_reset=True,NR_reward=0.),'Actual phase-changed tensors/origin differ')
    opt=rs['optimizer'];need(len(opt['param_groups'])==1,'One actual Adam group required');group=opt['param_groups'][0];ids=group['params']
    need(len(ids)==len(set(ids))==len(opt['state'])==m['optimizer_state_entries']==112 and set(ids)==set(opt['state']) and
         group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0 and group['foreach'] is False and
         abs(group['lr']-(2e-6+.5*(2e-5-2e-6)*(1+math.cos(math.pi*(N-1)/1199))))<1e-15,'Actual Adam count/ids/hyperparameters differ')
    for ident,v in zip(ids,ck['state'].values()):
        st=opt['state'][ident];need(int(st['step'])==N and all(st[k].dtype==torch.float32 and st[k].shape==v.shape and bool(torch.isfinite(st[k]).all()) for k in ['exp_avg','exp_avg_sq']),'Actual finite Adam moments/counters differ')
    need({'python_rng','numpy_rng','torch_rng','cuda_rng','grad_scaler'}<=set(rs) and m['all4_RNG_saved'] is True and
         m['grad_scaler_saved'] is True and m['all_FP32_finite'] is True and not torch.cuda.is_initialized(),'CPU registration/RNG/scaler contract differs')
    src=R/'source_payload/pixg5i_original/model_dir';adapter=Path(__file__).with_name('pixg5i_protocol_adapter.py')
    need(sha(adapter)==ADAPTER and all(sha(src/k)==h for k,h in FILES.items()),'Original deployment sources changed')
    label=f'pixg5i_hypir_structure_v2_s{N}';md=R/'submissions'/label/'model_dir';need(not md.parent.exists(),'Never overwrite a registered model')
    provenance=dict(config=ref(run/'config.json'),marker=ref(marker_path),input_catalog=ref(m['input_catalog']),teacher_catalog=ref(m['teacher_catalog']),resume=ref(resume),training_source=ref(source[0]))
    teacher=dict(raw_sha256=TEACHER,outputs=str(Path(q['teacher_manifest']).parent/'render/test'),applied_to_this_checkpoint=True,render_manifest=ref(q['teacher_manifest']))
    binding=dict(schema='pixg5i_hypir_distill_binding_v1',label=label,stage='distilled_checkpoint',checkpoint_sha256=m['raw_sha256'],
        initial_package_checkpoint_sha256=INITIAL,archive_sha256='f8e154da5ea9cb2fe165a316f824003459661945e6bbe22efd025d8592d573d6',
        training_provenance=provenance,teacher=teacher,used_as_G4_reference=False,variant='structure_v2',local_step=N,optimizer_step=N,reference_step=30000,
        training_phase='structure_v2',global_step=600+N,parent_distillation_steps=600,total_distillation_steps=600+N,
        phase_initial_checkpoint_sha256=PARENT,phase_parent=parent,parent_candidate_manifest=ref(parent_cm_path))
    proof=dict(status='CPU_ACTUAL_TRAINED_CHECKPOINT_VALIDATED',model_dir=str(md),binding=binding,actual_changed_FP32_tensors_vs_phase_parent=changed,
        actual112_raw_resume_equal_FP32_finite=True,actual112_Adam_moments_finite_and_counters=True,all4_RNG_saved=True,
        teacher_catalog_bound_to_fixed_manifest=True,teacher_training_PNG_bytes_rehashed_here=False,CPU_only=True,CUDA_initialized=False)
    if a.execute:
        md.parent.mkdir(exist_ok=False);md.mkdir()
        for old,new in [('runner.py','reference_runner.py'),('pix_model.py','pix_model.py'),('config.yaml','config.yaml')]:copy(src/old,md/new)
        copy(raw,md/'model.pth');copy(adapter,md/'runner.py');write(md/'pixg5i_binding.json',binding)
        cm=dict(schema='pixg5i_registered_candidate_v1',label=label,model_dir=str(md),stage='distilled_checkpoint',variant='structure_v2',
            model_type='supplied_pixG5i_pixel_UNet',checkpoint=str(raw),checkpoint_sha256=m['raw_sha256'],deployed_checkpoint=str(md/'model.pth'),
            initial_package_checkpoint_sha256=INITIAL,runner_sha256=ADAPTER,binding=ref(md/'pixg5i_binding.json'),files={x.name:ref(x) for x in md.iterdir() if x.is_file()},
            training_provenance=provenance,teacher=teacher,local_step=N,optimizer_step=N,reference_step=30000,
            training_phase='structure_v2',global_step=600+N,parent_distillation_steps=600,total_distillation_steps=600+N,
            phase_initial_checkpoint_sha256=PARENT,phase_parent=parent,parent_candidate_manifest=ref(parent_cm_path),
            registration_CPU_proof=proof,quality_reuse_claim=False,used_as_G4_reference=False,registration_source=ref(__file__),CPU_only=True,GPU_used=False,
            original_inference_source_unchanged=True)
        write(md.parent/'candidate_manifest.json',cm)
        proof=dict(status='REGISTERED_REAL_TRAINED_PIXG5I_CPU_ONLY',model_dir=str(md),label=label,candidate_manifest=ref(md.parent/'candidate_manifest.json'),checkpoint_sha256=m['raw_sha256'],GPU_used=False)
    print(json.dumps(proof,ensure_ascii=False))
if __name__=='__main__':main()
