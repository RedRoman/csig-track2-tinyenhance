"""PixG5i structure phase from actual medium600 and frozen HYPIR test100 PNGs.
No NR reward, validation data, teacher inference, architecture change or deployment mixing.
FP32 masters/Adam, FP16 autocast, raw RGB L1/gradient and clamped RGB LPIPS.
"""
from pathlib import Path
from collections import OrderedDict
import argparse, fcntl, hashlib, importlib.util, json, math, os, random, sys, time

R=Path('/cache/independent_sr_cuda_20261013')
INIT_SHA='f3781766ff4f635d6e903f3fa0641d0fb1c1211842721cc9185f5bf00c119487'
PIX_SHA='2f1cd3cd8311381e432138f489b01b982e08ebd5b09ecbc05ac87f7bfb577923'
RUNNER_SHA='6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'
TEACHER_SHA='23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c'
TM_SHA='f1a2004d3e4fa56272bc00183a84960510dcc6066bb98b0d29930d04a7642fd6'
TP_SHA='e83eb1eb0c0e04b6f2bd9ce7ac6c07d2662bc7173796792b0774b650d8802f11'
STATE_SHA='537960d2c9d5b6e1a728d8ed6017956a2901d375be72a23ce7fbb99f9756a36e'
LPIPS_WEIGHTS={'torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth':'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0',
 'torch/hub/checkpoints/alexnet-owt-7be5be79.pth':'7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'}

def need(v,m):
    if not v:raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def stamp(p):
    s=Path(p).stat();return (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)
def read(p):return json.loads(Path(p).read_text())
def atomic(p,data,torch=None):
    t=p.with_name(p.name+'.tmp.'+str(os.getpid()));need(not t.exists(),'Stale temporary file')
    with t.open('xb') as f:
        if torch is None:f.write((json.dumps(data,indent=2,allow_nan=False)+'\n').encode())
        else:torch.save(data,f)
        f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def load_source(p):
    s=importlib.util.spec_from_file_location('_pixg5i_original_architecture',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def parse():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--config',type=Path);k,_=pre.parse_known_args()
    cfg=read(k.config) if k.config else {};p=argparse.ArgumentParser(parents=[pre],description=__doc__)
    p.add_argument('--root',type=Path,default=R);p.add_argument('--run',choices=['pixg5i_hypir_structure_v2'],required=not bool(cfg))
    p.add_argument('--stop-at',type=int,default=100);p.add_argument('--resume',type=Path)
    p.add_argument('--stage-steps',type=int,default=1200);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--final-lr',type=float,default=2e-6)
    p.add_argument('--lpips-weight',type=float,default=.05);p.add_argument('--gradient-weight',type=float,default=.3)
    p.add_argument('--multiscale-l1-weight',type=float,default=.5);p.add_argument('--multiscale-gradient-weight',type=float,default=.3)
    p.add_argument('--seed',type=int,default=20261016);p.add_argument('--validate-config-only',action='store_true')
    need(not(set(cfg)-{x.dest for x in p._actions}),'Unknown CLI fields');p.set_defaults(**cfg);a=p.parse_args()
    need(a.run=='pixg5i_hypir_structure_v2' and a.stage_steps==1200 and 1<=a.stop_at<=1200,'Bounded independent structure run1200')
    need(a.lr==2e-5 and a.final_lr==2e-6 and a.seed==20261016,'Fixed structure-phase LR/seed')
    need((a.lpips_weight,a.gradient_weight,a.multiscale_l1_weight,a.multiscale_gradient_weight)==(.05,.3,.5,.3),'Fixed one structure objective')
    return a

def main():
    a=parse()
    if a.validate_config_only:print(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}));return
    root=Path(a.root).resolve();ex=root/'experiments/pixg5i_hypir_distill_20261015';out=ex/'runs'/a.run
    md=root/'source_payload/pixg5i_original/model_dir';teacher=root/'nr_push7_recovered_full_evaluations/push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all'
    tm=teacher/'render_manifest.json';tp=teacher/'render_protocol.json';scope=ex/'USER_SCOPE.json'
    parent_dir=ex/'runs/pixg5i_hypir_medium';parent_marker=parent_dir/'checkpoint_step600.json';parent_config=parent_dir/'config.json'
    pm=read(parent_marker);pc=read(parent_config)
    need(pm['local_step']==pm['optimizer_step']==600 and pm['protocol_id']==pc['protocol_id']=='f23c18ad89951a7a794e6bec2032f53d31e47d5c34659898d92a913bc453be05','Actual sealed medium600 parent')
    need(sha(parent_config)=='fc858e493f4b12ad10dd7005eff4b30401d63dc9a7ef120654c763973d926d71' and pc['protocol']['teacher_checkpoint_sha256']==TEACHER_SHA and pc['protocol']['initial_checkpoint_sha256']==INIT_SHA,'Exact parent original config/teacher')
    parent_raw=Path(pm['raw_checkpoint']);parent_resume=Path(pm['resume_checkpoint'])
    need(parent_raw==parent_dir/'step600.pth' and parent_resume==parent_dir/'resume_step600.pt' and sha(parent_raw)==pm['raw_sha256'] and sha(parent_resume)==pm['resume_sha256'],'Actual immutable parent raw/resume SHA')
    parent_sources=pc['protocol']['source_sha256']
    need(parent_sources[str(root/'code/pixg5i_hypir_distill/train_pixg5i_hypir.py')]=='d58e6dd66eac6b3f93a899a3008e48a568a9601a62b52cc3c99d73b9f1612604','Frozen d58 parent producer')
    for path,digest in parent_sources.items():need(sha(path)==digest,'Parent original producer changed')
    for p,h in [(md/'model.pth',INIT_SHA),(md/'pix_model.py',PIX_SHA),(md/'runner.py',RUNNER_SHA),(tm,TM_SHA),(tp,TP_SHA)]:need(sha(p)==h,'Frozen source/teacher changed: '+str(p))
    need(scope.is_file(),'Recorded new distillation authorization missing')
    manifest=read(tm);rp=read(tp);p=rp['protocol']
    need(manifest['scope']=='test100+validation3' and manifest['protocol']==p and manifest['protocol_id']==rp['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'Teacher protocol')
    ts=read(teacher/'render_status.json')
    need(p['runner']['checkpoint_sha256']==TEACHER_SHA and ts['status']=='complete' and ts['completed']==ts['total']==103 and (teacher/'render_complete').exists(),'Actual latest full teacher missing')
    records=[r for r in manifest['records'] if r['split']=='test'];tasks=[r for r in p['tasks'] if r['split']=='test']
    need([r['case'] for r in records]==[r['case'] for r in tasks]==list(range(1,101)),'Test100 only, no validation training')
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None;inputs=[];targets=[];guards={}
    for t,r in zip(tasks,records):
        src=root/'source_payload/assets/input'/f'case{t["case"]}.jpg';target=(teacher/r['output']).resolve()
        need(Path(t['input']).resolve()==src.resolve() and sha(src)==t['input_sha256'],'Actual test input SHA')
        need(target.parent==teacher/'render/test' and sha(target)==r['sha256'],'Actual teacher PNG SHA')
        with Image.open(src) as im,Image.open(target) as gt:
            need(im.size==gt.size==tuple(r['size']) and min(im.size)>=1024 and gt.mode=='RGB' and gt.format=='PNG','Native aligned teacher/input size')
        inputs.append(dict(case=t['case'],path=str(src),sha256=t['input_sha256'],size=r['size']))
        targets.append(dict(case=r['case'],path=str(target),sha256=r['sha256'],size=r['size'],input_sha256=t['input_sha256']))
        guards[str(src)]=stamp(src);guards[str(target)]=stamp(target)
    sources={str(x.resolve()):sha(x) for x in [Path(__file__),Path(a.config),md/'pix_model.py',md/'runner.py',md/'config.yaml',tm,tp,scope,parent_marker,parent_config]}
    for x in [md/'model.pth',parent_raw,parent_resume,*map(Path,sources)]:guards[str(x)]=stamp(x)
    import torch,numpy as np
    need(torch.cuda.is_available() and 'torch_npu' not in sys.modules,'Native CUDA required');torch.cuda.set_device(0);device=torch.device('cuda:0');torch.set_num_threads(4)
    def seed():random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    def rng():return dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0))
    seed();original=torch.load(md/'model.pth',map_location='cpu',weights_only=True)
    need(original['step']==30000 and len(original['state'])==112 and sum(v.numel() for v in original['state'].values())==3786764,'Actual PixG5i state schema')
    need(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in original['state'].values()),'Initial finite FP32')
    package_fingerprint=hashlib.sha256()
    for k,v in original['state'].items():package_fingerprint.update(json.dumps([k,list(v.shape),str(v.dtype)],separators=(',',':')).encode());package_fingerprint.update(b'\0');package_fingerprint.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    need(package_fingerprint.hexdigest()==STATE_SHA,'Original package fingerprint')
    warm=torch.load(parent_raw,map_location='cpu',weights_only=True);warm_resume=torch.load(parent_resume,map_location='cpu',weights_only=False)
    need(warm['cfg']==warm_resume['cfg']==original['cfg'] and warm['step']==warm_resume['local_step']==warm_resume['optimizer_step']==600 and warm['protocol_id']==warm_resume['protocol_id']==pm['protocol_id'],'Actual medium600 cfg/counters/protocol')
    need(list(warm['state'])==list(warm_resume['state'])==list(original['state']) and all(v.dtype==warm_resume['state'][k].dtype==torch.float32 and v.shape==original['state'][k].shape and bool(torch.isfinite(v).all()) and torch.equal(v,warm_resume['state'][k]) for k,v in warm['state'].items()),'All112 exact finite parent raw/resume FP32')
    need(len(warm_resume['optimizer']['state'])==112 and {int(v['step']) for v in warm_resume['optimizer']['state'].values()}=={600} and all(k in warm_resume for k in ['python_rng','numpy_rng','torch_rng','cuda_rng','grad_scaler']),'Actual old Adam600/fourRNG stored but deliberately NOT inherited')
    need(pc['protocol']['input_catalog']==inputs and pc['protocol']['teacher_catalog']==targets,'Exact original test100/teacher catalog')
    parent_binding=dict(marker=dict(path=str(parent_marker),sha256=sha(parent_marker)),config=dict(path=str(parent_config),sha256=sha(parent_config)),raw=dict(path=str(parent_raw),sha256=pm['raw_sha256']),resume=dict(path=str(parent_resume),sha256=pm['resume_sha256']),parent_local_step=600,parent_optimizer_step=600,raw_resume_FP32_exact=True,optimizer_inherited=False,RNG_inherited=False)
    pix=load_source(md/'pix_model.py');cfg=dict(original['cfg']);version=cfg.pop('arch_ver',1);need(version==2,'Original PixUNet2 only')
    student=pix.PixUNet2(**cfg);student.load_state_dict(warm['state'],strict=True);student=student.to(device,dtype=torch.float32,memory_format=torch.channels_last).eval().requires_grad_(True)
    initial={k:v.clone() for k,v in warm['state'].items()};names=[n for n,_ in student.named_parameters()];params=list(student.parameters())
    need(names==list(initial) and len(params)==112 and all(torch.equal(v.detach().cpu(),initial[k]) for k,v in student.state_dict().items()),'Exact medium600 FP32 masters')
    h=hashlib.sha256()
    for k,v in initial.items():h.update(json.dumps([k,list(v.shape),str(v.dtype)],separators=(',',':')).encode());h.update(b'\0');h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    initial_fingerprint=h.hexdigest()
    del warm_resume
    folded=pix.fold_inference(student).to(device,dtype=torch.float32,memory_format=torch.channels_last).eval().requires_grad_(False)
    weights=root/'scoring_tool/weights'
    for rel,digest in LPIPS_WEIGHTS.items():need(sha(weights/rel)==digest,'Frozen LPIPS weight changed')
    os.environ['TORCH_HOME']=str(weights/'torch');os.environ['HF_HUB_OFFLINE']='1';torch.hub.set_dir(str(weights/'torch/hub'))
    sys.path.insert(0,str(root/'code'));import numpy_legacy_compat
    import pyiqa
    lpips_metric=pyiqa.create_metric('lpips',device=device,as_loss=True,loss_reduction='none').eval().requires_grad_(False);lpips=lpips_metric.net
    lpips_source=Path(sys.modules[lpips.__class__.__module__].__file__).resolve();need(sha(lpips_source)=='7ee2ff201a4da4bdd4605ef0a6b539889c4ea4db4ba20a3c4f4831a1b43b1480','Original LPIPS source')
    sources[str(lpips_source)]=sha(lpips_source)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    flags=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    optimizer=torch.optim.Adam(params,lr=a.lr,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    protocol=dict(schema='pixg5i_HYPIR_structure_fidelity_v2',run=a.run,initial_checkpoint=str(md/'model.pth'),initial_checkpoint_sha256=INIT_SHA,
        initial_state_fingerprint_sha256=initial_fingerprint,parent=parent_binding,parent_distillation_steps=600,phase_initial_checkpoint_sha256=pm['raw_sha256'],reference_step=30000,architecture_cfg=original['cfg'],state_tensor_count=112,parameter_elements=3786764,
        origin='Structure continuation of supplied PixG5i medium600; fresh Adam0/new seed; not G4 and not HYPIR architecture',teacher_checkpoint_sha256=TEACHER_SHA,
        teacher_manifest=str(tm),teacher_manifest_sha256=TM_SHA,teacher_protocol_sha256=TP_SHA,teacher_scope='test100 frozen native PNGs; validation images/GT excluded',
        input_catalog=inputs,teacher_catalog=targets,source_sha256=sources,LPIPS_weights_sha256=LPIPS_WEIGHTS,
        stage_steps=1200,save_steps=[100,300,600,1200],seed=a.seed,optimizer='fresh FP32 Adam, step0, foreachFalse, no inherited moments',
        objective=dict(L1_teacher=1.,LPIPS_teacher=a.lpips_weight,gradient_teacher=a.gradient_weight,multiscale_L1_teacher=a.multiscale_l1_weight,multiscale_gradient_teacher=a.multiscale_gradient_weight,multiscale_factors=[2,4],NR_reward=0.,original_output_anchor=0.),
        loss_domain='raw_y01 native and avg-pool(2,4) L1/first differences; clamp01 LPIPS -> m11 normalizeFalse; no quantization/NR/GT',
        gradient_definition='0.5*(mean_abs(dx_student-dx_teacher)+mean_abs(dy_student-dy_teacher)) in RGB01 units',
        multiscale_definition='Average across factors2,4 of avg_pool2d RGB01 L1 and coarse-grid first-difference L1; no divide-by-scale physical-gradient normalization, phase aligned on origin8; no new target/NR/sharpening',
        lr_schedule=dict(kind='cosine',initial=a.lr,final=a.final_lr,total_steps=1200),input='RGB uint8/255 FP32 -> half()*2-1 -> FP32 for unfolded AMP time frequencies',
        training_precision='unfolded FP32 masters; FP16 autocast forward; FP32 loss/LPIPS; GradScaler fail-fast on nonfinite',
        deployment='Original FP32 fold then FP16 Runner unchanged; no teacher reads or mixing at inference',context=1024,loss_center=512,halo=256,crop_grid=8,
        data_cache='LRU2 decoded uint8 pairs; no inference output cache',flags=flags,no_val_GT_training=True,used_as_G4_reference=False)
    pid=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest();seed();done=0
    out.mkdir(parents=True,exist_ok=True)
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.resume:
            saved=torch.load(Path(a.resume),map_location='cpu',weights_only=False)
            need(Path(a.resume).resolve().parent==out and saved['protocol_id']==pid and saved['local_step']==saved['optimizer_step'],'Exact same-run resume')
            need(list(saved['state'])==names and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in saved['state'].values()),'Resume exact FP32 state required')
            need(len(saved['optimizer']['param_groups'])==1,'Resume one Adam group')
            ids=saved['optimizer']['param_groups'][0]['params'];need(len(ids)==112 and len(set(ids))==112,'Resume Adam parameter order')
            for ident,param in zip(ids,params):
                if saved['local_step']:
                    st=saved['optimizer']['state'][ident]
                    need(int(st['step'])==saved['local_step'] and all(st[k].dtype==torch.float32 and tuple(st[k].shape)==tuple(param.shape) and bool(torch.isfinite(st[k]).all()) for k in ['exp_avg','exp_avg_sq']),'Resume FP32 Adam shape/counter')
            student.load_state_dict(saved['state'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['grad_scaler']);done=saved['local_step']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],0)
            need(len(optimizer.state)==(112 if done else 0) and (not done or {int(v['step']) for v in optimizer.state.values()}=={done}),'Resume Adam counters')
        else:need(not (out/'config.json').exists(),'Existing run requires explicit resume')
        need(0<=done<a.stop_at<=1200,'Forward continuation required')
        old_rows=[readline for readline in (out/'train.jsonl').read_text().splitlines() if readline] if (out/'train.jsonl').exists() else []
        need([json.loads(x)['local_step'] for x in old_rows]==list(range(1,done+1)),'Resume must match complete current log')
        if (out/'config.json').exists():need(read(out/'config.json')['protocol_id']==pid,'Sealed protocol changed')
        else:
            atomic(out/'config.json',dict(protocol_id=pid,protocol=protocol));atomic(out/'input_catalog.json',inputs);atomic(out/'teacher_catalog.json',targets)
        def save(step):
            state={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
            need(list(state)==names and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()),'Saved112 FP32 state')
            need(len(optimizer.state)==(112 if step else 0) and (not step or {int(v['step']) for v in optimizer.state.values()}=={step}),'Actual Adam112/counter')
            for v in optimizer.state.values():
                for k in ['exp_avg','exp_avg_sq']:need(v[k].dtype==torch.float32 and bool(torch.isfinite(v[k]).all()),'Finite FP32 Adam moments')
            changed=sum(not torch.equal(v,initial[k]) for k,v in state.items());need(changed>0 if step else changed==0,'Actual new updates')
            raw=out/f'step{step}.pth';resume=out/f'resume_step{step}.pt';need(not raw.exists() and not resume.exists(),'Immutable milestone already exists')
            origin=dict(model_family='PixG5i_PixUNet2',initial_package_checkpoint_sha256=INIT_SHA,teacher_checkpoint_sha256=TEACHER_SHA,new_training_steps=step,parent_distillation_steps=600,total_distillation_steps=600+step,parent_checkpoint_sha256=pm['raw_sha256'],optimizer_reset=True,NR_reward=0.)
            atomic(raw,dict(cfg=original['cfg'],state=state,step=step,reference_step=30000,origin=origin,protocol_id=pid),torch)
            saved=dict(cfg=original['cfg'],state=state,optimizer=optimizer.state_dict(),grad_scaler=scaler.state_dict(),local_step=step,optimizer_step=step,protocol_id=pid,initial_checkpoint_sha256=INIT_SHA,phase_initial_checkpoint_sha256=pm['raw_sha256'],parent_distillation_steps=600,total_distillation_steps=600+step,**rng())
            atomic(out/'resume_latest.pt',saved,torch);os.link(out/'resume_latest.pt',resume)
            atomic(out/f'checkpoint_step{step}.json',dict(protocol_id=pid,local_step=step,optimizer_step=step,global_step=600+step,global_step_scope='total new distillation steps since supplied package; current phase Adam/local steps separate',parent_distillation_steps=600,phase_initial_checkpoint_sha256=pm['raw_sha256'],parent=parent_binding,reference_step=30000,
                raw_checkpoint=str(raw),raw_sha256=sha(raw),resume_checkpoint=str(resume),resume_sha256=sha(resume),source_sha256=sources,
                state_tensor_count=112,parameter_elements=3786764,changed_FP32_state_tensors=changed,all_FP32_finite=True,optimizer_state_entries=len(optimizer.state),all4_RNG_saved=True,grad_scaler_saved=True,
                initial_package_checkpoint_sha256=INIT_SHA,teacher_checkpoint_sha256=TEACHER_SHA,input_catalog=str(out/'input_catalog.json'),input_catalog_sha256=sha(out/'input_catalog.json'),teacher_catalog=str(out/'teacher_catalog.json'),teacher_catalog_sha256=sha(out/'teacher_catalog.json'),no_val_GT_training=True,NR_reward=0.))
        if done==0 and a.resume is None:save(0)
        if done:folded.half()
        cache=OrderedDict()
        def pair(case):
            for rec in [inputs[case-1],targets[case-1]]:need(stamp(rec['path'])==guards[rec['path']],'Input/teacher changed')
            if case not in cache:
                arrays=[]
                for rec in [inputs[case-1],targets[case-1]]:
                    with Image.open(rec['path']) as im:arrays.append(np.asarray(im.convert('RGB'),dtype=np.uint8).copy())
                cache[case]=arrays
                if len(cache)>2:cache.popitem(last=False)
            cache.move_to_end(case);return cache[case]
        def grad_loss(x,y):return .5*((x[:,:,:,1:]-x[:,:,:,:-1]-y[:,:,:,1:]+y[:,:,:,:-1]).abs().mean()+(x[:,:,1:,:]-x[:,:,:-1,:]-y[:,:,1:,:]+y[:,:,:-1,:]).abs().mean())
        start=time.time()
        with (out/'train.jsonl').open('a',buffering=1) as log:
            for step in range(done+1,a.stop_at+1):
                started=time.time();case=random.randint(1,100);lq,gt=pair(case);h,w=lq.shape[:2];left=8*random.randrange((w-1024)//8+1);top=8*random.randrange((h-1024)//8+1)
                x=torch.from_numpy(lq[top:top+1024,left:left+1024].copy()).permute(2,0,1)[None].to(device,dtype=torch.float32).div_(255)
                xm11=(x.half()*2-1).float().contiguous(memory_format=torch.channels_last)
                target=torch.from_numpy(gt[top+256:top+768,left+256:left+768].copy()).permute(2,0,1)[None].to(device,dtype=torch.float32).div_(255)
                optimizer.zero_grad(set_to_none=True)
                relation=None
                if step==1:
                    with torch.no_grad(),torch.amp.autocast('cuda',enabled=False):
                        u=student.core(xm11).clamp(-1,1);f=folded(xm11);difference=(u-f).abs();relation=dict(FP32_unfolded_vs_folded_max_abs=float(difference.max()),FP32_unfolded_vs_folded_mean_abs=float(difference.mean()))
                    folded.half()
                    with torch.no_grad():reference=folded(xm11.half()).float()
                with torch.amp.autocast('cuda',dtype=torch.float16):prediction=student.core(xm11)
                raw=(prediction.float()+1)*.5;rgb=raw[:,:,256:768,256:768];clipped=rgb.clamp(0,1)
                with torch.amp.autocast('cuda',enabled=False):
                    l1=(rgb-target).abs().mean();perceptual=lpips(clipped*2-1,target*2-1,normalize=False).mean();gradient=grad_loss(rgb,target)
                    coarse=[(torch.nn.functional.avg_pool2d(rgb,f,f),torch.nn.functional.avg_pool2d(target,f,f)) for f in [2,4]]
                    ms_l1=sum((sr-gt).abs().mean() for sr,gt in coarse)/2
                    ms_gradient=sum(grad_loss(sr,gt) for sr,gt in coarse)/2
                    need(all(bool(torch.isfinite(v)) for v in [l1,perceptual,gradient,ms_l1,ms_gradient]),'All five fidelity terms finite')
                    loss=l1+a.lpips_weight*perceptual+a.gradient_weight*gradient+a.multiscale_l1_weight*ms_l1+a.multiscale_gradient_weight*ms_gradient
                need(bool(torch.isfinite(loss)) and not target.requires_grad,'Finite teacher loss/detached PNG')
                if relation is not None:
                    diff=(prediction.detach().float().clamp(-1,1)-reference).abs();relation.update(AMP_unfolded_vs_initial_fold_FP16_max_abs=float(diff.max()),AMP_unfolded_vs_initial_fold_FP16_mean_abs=float(diff.mean()),comparison_scope='same medium600 initial weights/same FP16-quantized1024 input, no optimizer update; finite arithmetic difference recorded, no exact claim')
                    atomic(out/'initial_deployment_relation.json',relation)
                first=step==done+1
                if first:
                    g=torch.autograd.grad(loss,rgb,retain_graph=True)[0];need(bool(torch.isfinite(g).all()) and bool(g.abs().max()>0),'True finite RGB fidelity gradient')
                    component_RGB_gradients={}
                    for label,term in [('L1',l1),('LPIPS',a.lpips_weight*perceptual),('gradient',a.gradient_weight*gradient),('MS_L1',a.multiscale_l1_weight*ms_l1),('MS_gradient',a.multiscale_gradient_weight*ms_gradient)]:
                        cg=torch.autograd.grad(term,rgb,retain_graph=True)[0];need(bool(torch.isfinite(cg).all()),'Finite weighted component RGB gradient');component_RGB_gradients[label]=float(cg.float().norm())
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                finite=sum(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in params);nonzero=sum(v.grad is not None and bool(v.grad.abs().max()>0) for v in params)
                need(finite==112 and nonzero>0,'All112 gradients must exist/finite; AMP nonzero count recorded')
                norm=torch.nn.utils.clip_grad_norm_(params,1.);need(bool(torch.isfinite(norm)),'Finite unclipped gradient norm')
                need(not any(v.grad is not None for v in lpips.parameters()),'Frozen LPIPS received gradient')
                lr=a.final_lr+.5*(a.lr-a.final_lr)*(1+math.cos(math.pi*(step-1)/1199));optimizer.param_groups[0]['lr']=lr
                before=[v.detach().clone() for v in params];scaler.step(optimizer);scaler.update()
                changed=sum(not torch.equal(v.detach(),b) for v,b in zip(params,before));need(changed>0,'Real optimizer update required')
                update_sq=sum((v.detach()-b).float().square().sum() for v,b in zip(params,before));need(bool(torch.isfinite(update_sq)),'Finite actual update')
                torch.cuda.synchronize();row=dict(local_step=step,optimizer_step=step,case=case,context_crop_xy=[left,top],loss_crop_xy=[left+256,top+256],teacher_png_sha256=targets[case-1]['sha256'],
                    loss=float(loss.detach()),L1_teacher_raw=float(l1.detach()),LPIPS_teacher_clamped=float(perceptual.detach()),gradient_teacher_raw=float(gradient.detach()),multiscale_L1_teacher_raw=float(ms_l1.detach()),multiscale_gradient_teacher_raw=float(ms_gradient.detach()),global_distillation_step=600+step,
                    finite_parameter_gradients=finite,nonzero_parameter_gradients=nonzero,gradient_norm_before_clip=float(norm),actual_updated_tensors=changed,update_RMS=float((update_sq/3786764).sqrt()),lr=lr,grad_scaler_scale=scaler.get_scale(),
                    raw_outside01_fraction=float(((rgb.detach()<0)|(rgb.detach()>1)).float().mean()),step_seconds=time.time()-started,elapsed_seconds=time.time()-start)
                log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
                if first:atomic(out/f'gradient_preflight_step{step}.json',dict(status='PASS_FINITE112_AND_REAL_UPDATE',row=row,true_RGB_gradient_norm=float(g.float().norm()),FP32_master_all=True,frozen_LPIPS=True,teacher_detached=True,weighted_RGB_gradient_norms=component_RGB_gradients,phase_initial_checkpoint_sha256=pm['raw_sha256'],NR_reward=0.))
                atomic(out/'status.json',dict(stage='training',completed_local_steps=step,optimizer_step=step,target_steps=1200,row=row))
                if step in [100,300,600,1200] or step==a.stop_at:save(step)
                del x,xm11,target,prediction,raw,rgb,clipped,l1,perceptual,gradient,ms_l1,ms_gradient,coarse,loss,before,update_sq
        for path,token in guards.items():need(stamp(path)==token,'Frozen data/source changed during training')
        for path,digest in sources.items():need(sha(path)==digest,'Frozen source changed')
        atomic(out/'status.json',dict(stage='complete' if a.stop_at==1200 else 'stopped_for_review',completed_local_steps=a.stop_at,optimizer_step=a.stop_at,target_steps=1200,initial_package_checkpoint_sha256=INIT_SHA,teacher_checkpoint_sha256=TEACHER_SHA,NR_reward=0.,requires_actual_fresh_full103_418=True))

if __name__=='__main__':main()
