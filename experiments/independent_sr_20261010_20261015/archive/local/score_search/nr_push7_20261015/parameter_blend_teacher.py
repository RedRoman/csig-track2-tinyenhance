"""FP32 parameter interpolation within HYPIR; no training/distillation."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys
R=Path('/cache/independent_sr_cuda_20261013')
A=R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
N=R/'adaptation/runs/hypir_push7_A7750_niqe768_100/step7850.pth'
MARKER=N.parent/'checkpoint_step7850.json'
BASE=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
OUT=R/'adaptation/runs/hypir_push7_niqe768_parameter_blends'
ASHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
DSHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
def need(v,s):
 if not v:raise RuntimeError(s)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def js(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
if not args.execute:
 print(json.dumps(dict(status='CPU_PLAN_ONLY',alphas=[.25,.5],initial_A=str(A),finetuned=str(N),output=str(OUT),new_training=False,distillation=False)));sys.exit(0)
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(4)
need(not torch.cuda.is_initialized(),'CPU only')
need(not OUT.exists(),'Use new immutable derived directory')
m=json.loads(MARKER.read_text())
need(m['global_step']==7850 and m['local_step']==m['optimizer_step']==100 and m['no_teacher_outputs'] and m['no_val_GT'],'Real completed HYPIR NIQE100 required')
need(sha(A)==ASHA and sha(N)==m['raw_checkpoint_sha256'],'Actual input weights SHA changed')
need(sha(BASE/'runner.py')==DSHA,'Frozen runtime changed')
for p,h in m['source_sha256'].items():need(sha(p)==h,'Source changed '+p)
a=torch.load(A,map_location='cpu',weights_only=True,mmap=True);n=torch.load(N,map_location='cpu',weights_only=True,mmap=True)
need(list(a)==list(n) and len(a)==514 and sum(v.numel() for v in a.values())==259474432,'Original HYPIR514 schema')
need(all(a[k].dtype==n[k].dtype==torch.float32 and a[k].shape==n[k].shape and torch.isfinite(a[k]).all() and torch.isfinite(n[k]).all() for k in a),'Exact finite FP32 inputs required')
OUT.mkdir();results=[]
for tag,alpha in [('025',.25),('050',.5)]:
 label='push7_a7750_niqe100_alpha'+tag
 dest=R/'submissions'/('hypir_'+label)
 need(not dest.exists(),'Immutable candidate exists')
 state={k:torch.lerp(a[k],n[k],alpha) for k in a}
 need(all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values()),'Derived state nonfinite')
 changed=sum(not torch.equal(v,a[k]) for k,v in state.items())
 need(changed==514,'Expected all514 genuine parameter changes')
 raw=OUT/('alpha'+tag+'.pth')
 with raw.open('xb') as f:torch.save(state,f)
 rawsha=sha(raw)
 manifest=dict(schema='HYPIR_FP32_parameter_interpolation_v1',label=label,model_type='original_HYPIR_SD2_rank256_native512',alpha=alpha,recipe='torch.lerp(initial_A_FP32,NIQE100_FP32,alpha) on CPU per514 ordered tensors',checkpoint=str(raw),checkpoint_sha256=rawsha,initial_checkpoint=str(A),initial_checkpoint_sha256=ASHA,finetuned_checkpoint=str(N),finetuned_checkpoint_sha256=m['raw_checkpoint_sha256'],training_marker=dict(path=str(MARKER),sha256=sha(MARKER)),derivation_source_sha256=sha(__file__),FP32_tensor_count=514,elements=259474432,changed_tensors_from_A=changed,new_training=False,optimizer_state='not applicable: derived parameters, no new Adam/step/resume',distillation=False,lightweight_model=False,no_val_GT=True,no_teacher_outputs=True,quality_reuse_claim=False,CPU_only=True,CUDA_initialized=False)
 md=dest/'model_dir';md.mkdir(parents=True)
 shutil.copyfile(BASE/'runner.py',md/'runner.py');cfg=json.loads((BASE/'config.json').read_text());cfg.update(label=label,weight_path=str(raw));js(md/'config.json',cfg)
 manifest.update(model_dir=str(md),runner_sha256=DSHA,config_sha256=sha(md/'config.json'))
 js(dest/'candidate_manifest.json',manifest);results.append(manifest)
 del state
js(OUT/'derivation_manifest.json',dict(status='TWO_REAL_DERIVED_HYPIR_CHECKPOINTS',candidates=results,CUDA_initialized=torch.cuda.is_initialized(),script_sha256=sha(__file__)))
print(json.dumps(dict(status='COMPLETE',candidates=[dict(label=x['label'],raw=x['checkpoint'],sha256=x['checkpoint_sha256'],model_dir=x['model_dir'],alpha=x['alpha']) for x in results])),flush=True)

