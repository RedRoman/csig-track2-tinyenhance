"""Read-only CPU checks of actual lastconv50/100, logs and saved four RNG states."""
from pathlib import Path
import datetime,hashlib,json,math,os,random,statistics
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
from PIL import Image
torch.set_num_threads(4)
R=Path('/cache/independent_sr_cuda_20261013');D=R/'adaptation/runs/hypir_push7_A7750_lastconv_lr1e6_nr100'
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
need(sha(R/'code/push7/hypir_direct_nr/train_hypir_lastconv_nr_lr1e6_v2.py')=='8e8d5e1fcac90504d029c6df436172d9380d46ddc5b888dda01fd6b6528a25f7','Frozen trainer')
need(sha(R/'code/push7/hypir_direct_nr/config_hypir_lastconv_lr1e6_nr100_v2.json')=='65ca9462b4b3669c23d1f8333949066ae12864a540291a5e3eb8df9456144b7d','Frozen CLI')
for path,digest in p['source_sha256'].items():need(sha(path)==digest,'Actual source '+path)
need(p['teacher_output_loss'] is False and p['distillation'] is False and p['validation_GT_training'] is False and p['new_Adam_initial_step']==0,'FreshAdam/no teacher/noGT')
need(p['optimizer_tensor_count']==2 and p['requires_grad_LoRA_tensor_count']==514 and p['optimizer_parameter_keys']==list(KEYS),'Autograd514/optimizer2')
need(obj==dict(lr=1e-6,reward_weight=.1,musiq_weight=1.5,clip_weight=.1,input_l1=.02,input_lpips=.01,input_gradient=.05,parameter_anchor=1.,niqe_weight=.5),'Approved objective')
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
 need(finite_tree(r) and (r['global_step'],r['optimizer_step'])==(7750+i,i) and r['lr']==1e-6,'Finite log/counters')
 need(r['finite_optimizer_parameter_gradients']==r['nonzero_optimizer_parameter_gradients']==r['actual_updated_optimizer_tensors_this_step']==2 and r['all514_finite_gradient_count']==514,'Actual gradient/update counts')
 need(r['optimizer_tensor_count']==2 and r['requires_grad_LoRA_tensor_count']==514 and r['optimizer_parameter_keys']==list(KEYS),'Logged optimizer scope')
 content=.02*r['L1_input']+.01*r['LPIPS_input']+.05*r['gradient_input'];reward=1.5*r['scores']['musiq']+r['scores']['maniqa']+.1*r['scores']['clipiqa'];niqe=.1*.5*r['NIQE']/10
 need(abs(content-r['content'])<3e-7 and abs(reward-r['reward'])<5e-7 and abs(niqe-r['weighted_NIQE_penalty'])<1e-9,'Independent loss components')
 need(abs(content+r['parameter_anchor']-.1*reward+niqe-r['loss'])<5e-7,'Independent total loss')
 if i>1:need(abs(r['parameter_anchor']-rows[i-2]['update_mean_tensor_relative_squared'])<1e-9,'Previous true parameter anchor')
states={};milestones={}
for local in [0,50,100]:
 glob=7750+local;m=read(D/('checkpoint_step%d.json'%glob));saved=torch.load(Path(m['resume_checkpoint']),map_location='cpu',weights_only=False,mmap=True)
 need((m['global_step'],m['local_step'],m['optimizer_step'])==(glob,local,local) and m['protocol_id']==pid and m['source_sha256']==p['source_sha256'],'Marker counters/source')
 need((saved['global_step'],saved['local_step'],saved['optimizer_step'])==(glob,local,local) and saved['protocol_id']==pid and saved['optimizer_tensor_count']==2 and saved['optimizer_parameter_keys']==list(KEYS),'Saved counters/optimizer scope')
 need(len(saved['optimizer']['state'])==(2 if local else 0),'Fresh Adam2 counter states')
 random.Random().setstate(saved['python_rng']);np.random.RandomState().set_state(saved['numpy_rng']);torch.Generator(device='cpu').set_state(saved['torch_rng'])
 for key in ['torch_rng','cuda_rng']:need(saved[key].device.type=='cpu' and saved[key].dtype==torch.uint8 and saved[key].ndim==1 and saved[key].numel()>=16,'Actual RNG byte state')
 states[local]={k:saved[k] for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']}
 if local:
  cm=read(R/('submissions/hypir_push7_a7750_lastconv_lr1e6_nr_s%d/candidate_manifest.json'%glob))
  need(cm['registrar_source_sha256']=='c7bb01c0d9b0cc28fd972e1b4bfd60858116baef1337b0f9a936f680ee986a32' and cm['checkpoint_sha256']==m['raw_checkpoint_sha256'] and cm['actual_unselected512_equal_A'] is True and cm['actual_Adam_state_count']==2 and cm['local_step']==cm['optimizer_step']==local,'Actual root-registered lineage')
 milestones[str(local)]=dict(global_step=glob,optimizer_step=local,saved_RNG_valid=True,root_registration_exact_raw_resume_Adam2_frozen512_scope_referenced=bool(local),raw_sha256=m['raw_checkpoint_sha256'],resume_sha256=m['resume_checkpoint_sha256'],large_weights_rehashed_in_this_log_RNG_check=False)
 del saved
py=random.Random();py.setstate(states[0]['python_rng']);nr=np.random.RandomState();nr.set_state(states[0]['numpy_rng'])
for i,r in enumerate(rows,1):
 case=py.randint(1,100);w,h=sizes[case];crop=[py.randint(0,(w-768)//256)*256,py.randint(0,(h-768)//256)*256];need((case,crop)==(r['case'],r['crop_xy']),'Independent all100 case/crop replay')
 for _ in range(20):nr.randint(0,768-224+1);nr.randint(0,768-224+1)
 if i in states:need(py.getstate()==states[i]['python_rng'] and np_equal(nr.get_state(),states[i]['numpy_rng']),'Python and MANIQA20 NumPy milestone RNG exact')
need(all(torch.equal(states[0]['torch_rng'],s['torch_rng']) for s in states.values()),'Torch CPU RNG unchanged')
offsets={k:int.from_bytes(v['cuda_rng'][8:16].numpy().tobytes(),'little') for k,v in states.items()};seeds={v['cuda_rng'][:8].numpy().tobytes() for v in states.values()};need((offsets[50]-offsets[0])%50==0,'Uniform saved CUDA offset');increment=(offsets[50]-offsets[0])//50
need(len(seeds)==1 and increment>0 and offsets[100]==offsets[0]+100*increment,'Saved CUDA Philox continuity, not GPU replay')
need(not torch.cuda.is_initialized(),'No CUDA initialization')
result=dict(status='PASS_LOG100_RNG_AND_FROZEN_SOURCE',schema='lastconv_lr1e6_minimal_CPU_log_RNG_v1',run=str(D),source_sha256='8e8d5e1fcac90504d029c6df436172d9380d46ddc5b888dda01fd6b6528a25f7',CLI_sha256='65ca9462b4b3669c23d1f8333949066ae12864a540291a5e3eb8df9456144b7d',protocol_id=pid,updates=100,logs100_finite_counters_loss_recomputed=True,milestones=milestones,RNG=dict(all_four_saved_valid=True,Python100_case_crop_replay_exact=True,MANIQA20_NumPy100_replay_exact=True,TorchCPU_exact_unchanged=True,CUDA_saved_offsets=offsets,CUDA_replay_executed=False),CPU_only=True,CUDA_initialized=False,large_checkpoint_or_metric_weights_rehashed_here=False,raw_resume_semantic_scope='Referenced root CPU c7 registrar for exact514 raw/resume/selected2/512A/Adam2; capsule separately verifies complete original file SHA',source_or_run_files_modified=False,quality_conclusion=False,auditor_sha256=sha(__file__),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
print(json.dumps(result,indent=2),flush=True)
