"""Diagnostic forward comparison only; never a benchmark, full score, or quality reuse.
Run reference/candidate in separate fresh GPU processes selected by root.
Writes only a fresh --out-dir. Native suite saves about200MiB temporary FP32 reference data.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,os,pickle,random,time
import numpy as np
import torch
from PIL import Image
R=Path('/cache/independent_sr_cuda_20261013')
PARENT=R/'submissions/hypir_push4_a7750_compiled/model_dir'
PARENT_SOURCE='396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae'
WEIGHT_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def rh(s):
 return {'python':hashlib.sha256(pickle.dumps(s['python'],protocol=4)).hexdigest(),'numpy':hashlib.sha256(pickle.dumps(s['numpy'],protocol=4)).hexdigest(),'cpu':hashlib.sha256(s['cpu'].cpu().numpy().tobytes()).hexdigest(),'cuda':[hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in s['cuda']]}
def restore(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])
def png_digest(cpu):
 a=((cpu[0].permute(1,2,0).float().clamp(-1,1).numpy()*.5+.5)*255).round().astype(np.uint8)
 b=io.BytesIO();Image.fromarray(a).save(b,format='PNG',compress_level=1);return hashlib.sha256(b.getbuffer()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['reference','candidate'],required=True);p.add_argument('--suite',choices=['smoke512','native7_41'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
 d=a.out_dir;d.mkdir(parents=True,exist_ok=True);model=a.model_dir.resolve(strict=True);config=json.loads((model/'config.json').read_text());assert sha(config['weight_path'])==WEIGHT_SHA
 if a.phase=='reference':assert model==PARENT and sha(model/'runner.py')==PARENT_SOURCE
 else:
  manifest=json.loads((model.parent/'candidate_manifest.json').read_text());assert sha(model/'runner.py')==manifest['runner_sha256'] and sha(model/'config.json')==manifest['config_sha256'] and manifest['checkpoint_sha256']==WEIGHT_SHA
 initial=d/(a.suite+'_initial_rng.pt');result=d/(a.suite+'_'+a.phase+('_'+model.parent.name if a.phase=='candidate' else '')+'.json');assert not result.exists(),'Immutable diagnostic output exists'
 if a.phase=='reference':assert not initial.exists()
 spec=importlib.util.spec_from_file_location('runner',model/'runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);os.chdir(model);r=mod.Runner(str(model));torch.cuda.synchronize(r.device)
 state=rng()
 if a.phase=='reference':torch.save(state,initial)
 else:
  saved=torch.load(initial,map_location='cpu',weights_only=False);assert rh(saved)==rh(state),'Constructor four-RNG state differs';restore(saved)
 records=[];held=[];ownership=True
 keys=list(range(3)) if a.suite=='smoke512' else [7,41]
 with torch.no_grad():
  for key in keys:
   if a.suite=='smoke512':
    v=torch.arange(3*512*512,device=r.device,dtype=torch.float32).reshape(1,3,512,512)
    x=((v.remainder(256)/255).half()*2-1) if key==0 else (((v.flip(-1).remainder(251)/250).half()*2-1) if key==1 else torch.sin(v/17).half())
    source='three distinct deterministic512 tensors; no global RNG input generation'
   else:
    src=R/'source_payload/assets/input'/f'case{key}.jpg'
    with Image.open(src) as im:array=np.asarray(im.convert('RGB'),dtype=np.float32)/255.0
    x=torch.from_numpy(array).permute(2,0,1)[None].to(r.device).half()*2-1;source=str(src)
   input_sha=hashlib.sha256(x.float().cpu().numpy().tobytes()).hexdigest();began=time.monotonic();y=r.infer(x);torch.cuda.synchronize(r.device)
   assert y.dtype==torch.float32 and y.device==x.device and y.shape==x.shape
   cpu=y.detach().cpu().clone();assert bool(torch.isfinite(cpu).all()) and float(cpu.min())>=-1 and float(cpu.max())<=1
   for live,old in held:
    ownership=ownership and torch.equal(live.detach().cpu(),old)
   assert ownership,'Previously returned tensor was overwritten by later infer'
   held.append((y,cpu));state=rng();ref=d/f'{a.suite}_reference_{key}.pt';row={'key':key,'input':source,'input_float_sha256':input_sha,'shape':list(x.shape),'finite_and_in_range':True,'seconds_diagnostic_not_latency':time.monotonic()-began,'float_sha256':hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),'png_sha256':png_digest(cpu),'post_rng_sha256':rh(state)}
   if a.phase=='reference':assert not ref.exists();torch.save({'y':cpu,'rng':state,'input_sha256':input_sha,'png_sha256':row['png_sha256']},ref)
   else:
    saved=torch.load(ref,map_location='cpu',weights_only=False);assert saved['input_sha256']==input_sha
    delta=(cpu-saved['y']).abs();row.update(float_equal=torch.equal(cpu,saved['y']),max_abs=float(delta.max()),mean_abs=float(delta.mean()),png_equal=row['png_sha256']==saved['png_sha256'],post_four_rng_equal=rh(state)==rh(saved['rng']))
   records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='post_rng_sha256'}),flush=True)
 report={'status':'complete','scope':'diagnostic only; NOT full quality or benchmark, never sufficient alone for quality reuse','phase':a.phase,'suite':a.suite,'physical_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'model_dir':str(model),'checkpoint_sha256':WEIGHT_SHA,'runner_sha256':sha(model/'runner.py'),'helper_sha256':sha(__file__),'runner_protocol':r.protocol,'output_ownership_preserved':ownership,'rows':records}
 if a.phase=='candidate':report['all_float_png_four_rng_exact']=all(z['float_equal'] and z['png_equal'] and z['post_four_rng_equal'] for z in records)
 result.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'result':str(result),'status':report['status'],'all_float_png_four_rng_exact':report.get('all_float_png_four_rng_exact'),'output_ownership_preserved':ownership}),flush=True)
if __name__=='__main__':main()
