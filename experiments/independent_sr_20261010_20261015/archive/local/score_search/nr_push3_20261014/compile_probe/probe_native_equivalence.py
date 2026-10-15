"""Compare original/reduce-overhead on native case7 then41, separate processes.
This is a two-image diagnostic sequence, not the official val3+test100 sequence.
Reference saves its FP32 outputs and initial/post-call4RNG to /cache only.
Candidate compares floats, encodedPNG bytes and4RNG; it never modifies runners.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, io, json, os, random, time
import numpy as np
import torch
from PIL import Image
ROOT=Path('/cache/independent_sr_cuda_20261013')
REF=ROOT/'submissions/hypir_cuda_nr_stage3_anchor6250_m15_s6500_compiled/model_dir'
VAR=ROOT/'submissions/hypir_push3_6500_reduceoverhead_probe/model_dir'
SHAS={'reference':'396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae','candidate':'74f138aa24be01a7075a5a45b469a62ae70c9fe2dedc10c705a41a784e2c2d12'}
CHECKPOINT='cfcc3013fa87aa721f346d6cc285e70b6d6ff2c88315ce9f1465b07ffe020e68'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}
def restore(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch_cpu']);torch.cuda.set_rng_state_all(s['torch_cuda'])
def canonical(s):
 if torch.is_tensor(s):return [str(s.dtype),list(s.shape),hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest()]
 if isinstance(s,np.ndarray):return [str(s.dtype),list(s.shape),hashlib.sha256(s.tobytes()).hexdigest()]
 if isinstance(s,dict):return {k:canonical(v) for k,v in s.items()}
 if isinstance(s,(tuple,list)):return [canonical(v) for v in s]
 return s
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['reference','candidate'],required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();d=a.out_dir;d.mkdir(parents=True,exist_ok=True)
model=REF if a.phase=='reference' else VAR;config=json.loads((model/'config.json').read_text());assert sha(model/'runner.py')==SHAS[a.phase];assert sha(config['weight_path'])==CHECKPOINT
spec=importlib.util.spec_from_file_location('runner',model/'runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);os.chdir(model);runner=mod.Runner(str(model));torch.cuda.synchronize()
if a.phase=='reference':torch.save(rng(),d/'native_initial_rng.pt')
else:restore(torch.load(d/'native_initial_rng.pt',map_location='cpu',weights_only=False))
rows=[]
with torch.no_grad():
 for case in [7,41]:
  src=ROOT/'source_payload/assets/input'/f'case{case}.jpg'
  with Image.open(src) as im:array=np.asarray(im.convert('RGB'),dtype=np.float32)/255
  x=torch.from_numpy(array).permute(2,0,1)[None].to('cuda').half()*2-1
  began=time.monotonic();y=runner.infer(x).float().clamp(-1,1);torch.cuda.synchronize();cpu=y.detach().cpu().clone();assert cpu.shape==x.shape and bool(torch.isfinite(cpu).all());state=rng()
  png=io.BytesIO();Image.fromarray(((cpu[0].permute(1,2,0).numpy()*.5+.5)*255).round().astype(np.uint8)).save(png,format='PNG',compress_level=1)
  row={'case':case,'input_path':str(src),'input_sha256':sha(src),'native_shape':list(x.shape),'finite':True,'seconds':time.monotonic()-began,'output_float_sha256':hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),'output_png_sha256':hashlib.sha256(png.getbuffer()).hexdigest(),'output_png_bytes':png.getbuffer().nbytes,'post_rng':canonical(state)}
  if a.phase=='reference':torch.save({'y':cpu,'rng':state,'png_sha256':row['output_png_sha256']},d/f'native_reference_case{case}.pt')
  else:
   saved=torch.load(d/f'native_reference_case{case}.pt',map_location='cpu',weights_only=False);delta=(cpu-saved['y']).abs();row.update(output_float_equal=torch.equal(cpu,saved['y']),max_abs=float(delta.max()),mean_abs=float(delta.mean()),post_rng_equal=canonical(state)==canonical(saved['rng']),output_png_equal=row['output_png_sha256']==saved['png_sha256'])
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='post_rng'}),flush=True);del x,y,cpu,array;torch.cuda.empty_cache()
result={'status':'complete','phase':a.phase,'gpu_physical_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'checkpoint_sha256':CHECKPOINT,'runner_sha256':SHAS[a.phase],'helper_sha256':sha(__file__),'runner_protocol':runner.protocol,'sequence':[7,41],'scope':'two native images only, fresh diagnostic sequence; not formal quality scoring','rows':rows}
if a.phase=='candidate':result['both_native_cases_exact']=all(r['output_float_equal'] and r['post_rng_equal'] and r['output_png_equal'] for r in rows)
(d/f'native_{a.phase}_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
