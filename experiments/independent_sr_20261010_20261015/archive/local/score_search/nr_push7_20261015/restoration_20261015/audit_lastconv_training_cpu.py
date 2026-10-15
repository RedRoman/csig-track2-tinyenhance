"""Read-only CPU checks of actual lastconv50/100, logs and saved four RNG states."""
from pathlib import Path
import datetime,hashlib,json,math,os,random,statistics
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
from PIL import Image
torch.set_num_threads(4)
R=Path('/cache/independent_sr_cuda_20261013');D=R/'adaptation/runs/hypir_push7_A7750_lastconv_nr100'
KEYS=('conv_out.lora_A.default.weight','conv_out.lora_B.default.weight');ELEMENTS=738304
def need(v,m):
 if not v:raise RuntimeError(m)
HASHES={}
def sha(p):
 p=Path(p);s=p.stat();stamp=(s.st_size,s.st_mtime_ns,s.st_ino)
 if p in HASHES:
  need(HASHES[p][0]==stamp,'Source changed during audit');return HASHES[p][1]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 HASHES[p]=(stamp,h.hexdigest());return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def np_equal(a,b):return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
need(not torch.cuda.is_initialized(),'CPU only')
status=read(D/'status.json');need(status['stage']=='complete' and status['completed_local_steps']==status['optimizer_step']==100,'Wait for actual complete100')
cfg=read(D/'config.json');p=cfg['protocol'];obj=p['objective'];pid=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
need(pid==cfg['protocol_id'] and p['schema']=='hypir_push7_lastconv_nr_niqe768_optimizer2_full_gaussian_fused_bf16_v1','Protocol identity')
need(sha(R/'code/push7/hypir_direct_nr/train_hypir_lastconv_nr_v1.py')=='24dd2608888bd240b310d3a7fde678e376483229fe35c0cd7e71ab61e87f1082','Frozen trainer')
need(sha(R/'code/push7/hypir_direct_nr/config_hypir_lastconv_nr100_v1.json')=='547a4ccb4d064ba37a67456293bbd29d94f58441f5cf70b627cd27c33dd6d82e','Frozen CLI')
for path,digest in p['source_sha256'].items():need(sha(path)==digest,'Actual source '+path)
need(p['teacher_output_loss'] is False and p['distillation'] is False and p['validation_GT_training'] is False and p['new_Adam_initial_step']==0,'FreshAdam/no teacher/noGT')
need(p['optimizer_tensor_count']==2 and p['requires_grad_LoRA_tensor_count']==514 and p['optimizer_parameter_keys']==list(KEYS),'Autograd514/optimizer2')
need(obj==dict(lr=2e-7,reward_weight=.1,musiq_weight=1.5,clip_weight=.1,input_l1=.02,input_lpips=.01,input_gradient=.05,parameter_anchor=1.,niqe_weight=.5),'Approved objective')
inputs=read(D/'input_catalog.json');need(inputs==p['inputs'] and [x['case'] for x in inputs]==list(range(1,101)),'Test100 catalog')
sizes={}
for x in inputs:
 src=R/'source_payload/assets/input'/('case%d.jpg'%x['case']);need(Path(x['path']).resolve()==src.resolve() and sha(src)==x['sha256'] and not ({'teacher','gt','target'}&set(x)),'Only actual original100 LQ')
 with Image.open(src) as im:sizes[x['case']]=im.size
rows=[json.loads(x) for x in (D/'train.jsonl').read_text().splitlines() if x.strip()];need([x['local_step'] for x in rows]==list(range(1,101)),'100 real unique update rows')
def finite_tree(x):
 if isinstance(x,dict):return all(finite_tree(v) for v in x.values())
 if isinstance(x,list):return all(finite_tree(v) for v in x)
 return not isinstance(x,(int,float)) or math.isfinite(x)
for i,r in enumerate(rows,1):
 need(finite_tree(r) and (r['global_step'],r['optimizer_step'])==(7750+i,i) and r['lr']==2e-7,'Finite log/counters')
 need(r['finite_optimizer_parameter_gradients']==r['nonzero_optimizer_parameter_gradients']==r['actual_updated_optimizer_tensors_this_step']==2 and r['all514_finite_gradient_count']==514,'Actual gradient/update counts')
 need(r['optimizer_tensor_count']==2 and r['requires_grad_LoRA_tensor_count']==514 and r['optimizer_parameter_keys']==list(KEYS),'Logged optimizer scope')
 content=.02*r['L1_input']+.01*r['LPIPS_input']+.05*r['gradient_input'];reward=1.5*r['scores']['musiq']+r['scores']['maniqa']+.1*r['scores']['clipiqa'];niqe=.1*.5*r['NIQE']/10
 need(abs(content-r['content'])<3e-7 and abs(reward-r['reward'])<5e-7 and abs(niqe-r['weighted_NIQE_penalty'])<1e-9,'Independent loss components')
 need(abs(content+r['parameter_anchor']-.1*reward+niqe-r['loss'])<5e-7,'Independent total loss')
 if i>1:need(abs(r['parameter_anchor']-rows[i-2]['update_mean_tensor_relative_squared'])<1e-9,'Previous true parameter anchor')
initial_path=Path(p['parent_raw']);need(sha(initial_path)=='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d','Actual A7750 bytes')
initial=torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True);states={};milestones={}
for local in [0,50,100]:
 glob=7750+local;m=read(D/('checkpoint_step%d.json'%glob));rp=Path(m['raw_checkpoint']);sp=Path(m['resume_checkpoint'])
 need((m['global_step'],m['local_step'],m['optimizer_step'])==(glob,local,local) and m['protocol_id']==pid and m['source_sha256']==p['source_sha256'],'Actual marker counters/provenance')
 need(sha(rp)==m['raw_checkpoint_sha256'] and sha(sp)==m['resume_checkpoint_sha256'],'Immutable actual raw/resume SHA')
 raw=torch.load(rp,map_location='cpu',weights_only=True,mmap=True);saved=torch.load(sp,map_location='cpu',weights_only=False,mmap=True)
 need(list(raw)==list(saved['lora'])==list(initial) and len(raw)==514 and sum(v.numel() for v in raw.values())==259474432,'Raw514 schema')
 need(all(v.dtype==saved['lora'][n].dtype==torch.float32 and v.shape==saved['lora'][n].shape and bool(torch.isfinite(v).all()) and bool(torch.isfinite(saved['lora'][n]).all()) and torch.equal(v,saved['lora'][n]) for n,v in raw.items()),'Actual514 FP32 finite raw==resume')
 changed=[n for n,v in raw.items() if not torch.equal(v,initial[n])];need(changed==(list(KEYS) if local else []),'Only two changed / other512 exact A')
 need(m['unselected512_exact_A_count']==512 and m['unselected512_exact_A'] is True and m['changed_FP32_LoRA_tensors']==(2 if local else 0),'Marker actual scope')
 need((saved['global_step'],saved['local_step'],saved['optimizer_step'])==(glob,local,local) and saved['protocol_id']==pid and saved['optimizer_parameter_keys']==list(KEYS) and saved['optimizer_tensor_count']==2,'Saved counters/scope')
 opt=saved['optimizer'];need(len(opt['param_groups'])==1,'One Adam group');g=opt['param_groups'][0];ids=g['params']
 need(len(ids)==len(set(ids))==2 and g['lr']==2e-7 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 and g['foreach'] is False,'Fresh Adam options')
 if local:
  need(set(ids)==set(opt['state']),'Exactly two Adam states')
  for ident,key in zip(ids,KEYS):
   st=opt['state'][ident];need(float(st['step'])==local and bool(torch.isfinite(st['step']).all()),'Actual Adam counters')
   need(all(st[k].dtype==torch.float32 and st[k].device.type=='cpu' and st[k].shape==raw[key].shape and bool(torch.isfinite(st[k]).all()) for k in ['exp_avg','exp_avg_sq']) and bool((st['exp_avg_sq']>=0).all()),'Actual selected Adam moments')
 else:need(not opt['state'],'Fresh Adam0')
 random.Random().setstate(saved['python_rng']);np.random.RandomState().set_state(saved['numpy_rng']);torch.Generator(device='cpu').set_state(saved['torch_rng'])
 for key in ['torch_rng','cuda_rng']:need(saved[key].device.type=='cpu' and saved[key].dtype==torch.uint8 and saved[key].ndim==1 and saved[key].numel()>=16,'Actual RNG byte state')
 states[local]={k:saved[k] for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']}
 milestones[str(local)]=dict(global_step=glob,optimizer_step=local,raw514_FP32_finite_equal_resume=True,selected_changed=changed,other512_exact_A=True,Adam_states=len(opt['state']),Adam_moments_finite_FP32_CPU=True,raw_sha256=m['raw_checkpoint_sha256'],resume_sha256=m['resume_checkpoint_sha256'])
 if local==100:
  delta=[float((raw[k].double()-initial[k].double()).square().sum()) for k in KEYS];den=[float(initial[k].double().square().sum()) for k in KEYS];rms=math.sqrt(sum(delta)/ELEMENTS);relative=math.sqrt(sum(delta)/sum(den))
  need(abs(rms-rows[-1]['update_trainable_RMS_from_initial'])<1e-10 and abs(relative-rows[-1]['update_relative_L2_from_initial'])<1e-7,'Independent actual selected update')
 del raw,saved,opt
py=random.Random();py.setstate(states[0]['python_rng']);nr=np.random.RandomState();nr.set_state(states[0]['numpy_rng'])
for i,r in enumerate(rows,1):
 case=py.randint(1,100);w,h=sizes[case];crop=[py.randint(0,(w-768)//256)*256,py.randint(0,(h-768)//256)*256];need((case,crop)==(r['case'],r['crop_xy']),'Independent all100 case/crop replay')
 for _ in range(20):nr.randint(0,768-224+1);nr.randint(0,768-224+1)
 if i in states:need(py.getstate()==states[i]['python_rng'] and np_equal(nr.get_state(),states[i]['numpy_rng']),'Python and MANIQA20 NumPy milestone RNG exact')
need(all(torch.equal(states[0]['torch_rng'],s['torch_rng']) for s in states.values()),'Torch CPU RNG unchanged')
offsets={k:int.from_bytes(v['cuda_rng'][8:16].numpy().tobytes(),'little') for k,v in states.items()};seeds={v['cuda_rng'][:8].numpy().tobytes() for v in states.values()};need((offsets[50]-offsets[0])%50==0,'Uniform saved CUDA offset');increment=(offsets[50]-offsets[0])//50
need(len(seeds)==1 and increment>0 and offsets[100]==offsets[0]+100*increment,'Saved CUDA Philox continuity, not GPU replay')
need(not torch.cuda.is_initialized(),'No CUDA initialization')
result=dict(status='PASS',schema='lastconv_nr100_independent_CPU_actual_artifacts_v1',run=str(D),source_sha256='24dd2608888bd240b310d3a7fde678e376483229fe35c0cd7e71ab61e87f1082',CLI_sha256='547a4ccb4d064ba37a67456293bbd29d94f58441f5cf70b627cd27c33dd6d82e',protocol_id=pid,updates=100,logs100_finite_counters_loss_recomputed=True,test100_actual_input_SHA_checked=True,milestones=milestones,final_selected_update_RMS=rms,final_selected_relative_L2=relative,RNG=dict(all_four_saved_valid=True,Python100_case_crop_replay_exact=True,MANIQA20_NumPy100_replay_exact=True,TorchCPU_exact_unchanged=True,CUDA_saved_offset_per_step=increment,CUDA_saved_offsets=offsets,CUDA_replay_executed=False),CPU_only=True,CUDA_initialized=False,source_or_run_files_modified=False,median_step_seconds=statistics.median(x['step_seconds'] for x in rows[1:]),quality_conclusion=False,auditor_sha256=sha(__file__),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
print(json.dumps(result,indent=2),flush=True)
