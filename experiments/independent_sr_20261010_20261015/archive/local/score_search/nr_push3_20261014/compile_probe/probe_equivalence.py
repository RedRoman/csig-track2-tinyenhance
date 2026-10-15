from pathlib import Path
import argparse, hashlib, importlib.util, json, os, random, sys, time
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--reference',action='store_true');a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('runner',a.model_dir/'runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);os.chdir(a.model_dir);runner=mod.Runner(str(a.model_dir));torch.cuda.synchronize()
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}
def restore(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu']);torch.cuda.set_rng_state_all(s['torch_cuda'])
def canonical(s):
 if torch.is_tensor(s):return [str(s.dtype),list(s.shape),hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest()]
 if isinstance(s,np.ndarray):return [str(s.dtype),list(s.shape),hashlib.sha256(s.tobytes()).hexdigest()]
 if isinstance(s,dict):return {k:canonical(v) for k,v in s.items()}
 if isinstance(s,(tuple,list)):return [canonical(v) for v in s]
 return s
if a.reference:torch.save(rng(),a.out_dir/'initial_rng.pt')
else:restore(torch.load(a.out_dir/'initial_rng.pt',map_location='cpu',weights_only=False))
x=(torch.arange(3*512*512,device='cuda',dtype=torch.float32).remainder(256).reshape(1,3,512,512)/255).half()*2-1
rows=[]
with torch.no_grad():
 for i in range(3):
  y=runner.infer(x);torch.cuda.synchronize();cpu=y.detach().float().cpu().clone();assert cpu.shape==x.shape and bool(torch.isfinite(cpu).all());state=rng();row={'index':i,'all_finite':True,'output_sha256':hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),'post_rng':canonical(state)}
  if a.reference:torch.save({'y':cpu,'rng':state},a.out_dir/f'reference_{i}.pt')
  else:
   ref=torch.load(a.out_dir/f'reference_{i}.pt',map_location='cpu',weights_only=False);row.update(output_equal=torch.equal(cpu,ref['y']),max_abs=float((cpu-ref['y']).abs().max()),post_rng_equal=canonical(state)==canonical(ref['rng']))
  rows.append(row);print(json.dumps({'index':i,'phase':'reference' if a.reference else 'candidate','finite':True,**({} if a.reference else {k:row[k] for k in ['output_equal','max_abs','post_rng_equal']})}),flush=True)
result={'status':'complete','runner_protocol':runner.protocol,'rows':rows,'scope':'three consecutive512 calls only; not native4K/quality proof'}
if not a.reference:result['all_three_exact']=all(r['output_equal'] and r['post_rng_equal'] for r in rows)
(a.out_dir/('reference_smoke.json' if a.reference else 'candidate_smoke.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
