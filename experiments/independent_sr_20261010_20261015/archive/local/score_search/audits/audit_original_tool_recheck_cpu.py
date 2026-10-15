"""CPU-only integrity/418-scalar comparison of fresh original score_dir repeats."""
from pathlib import Path
import datetime
import hashlib
import json
import math
from PIL import Image

ROOT=Path('/cache/independent_sr_cuda_20261013')
RECHECK=ROOT/'metric_rechecks/20261014_original_tool'
PAIRS={
 'compiled6000':ROOT/'evaluations/s6000_mt200_ct175_compiled_cuda_all',
 'conservative6250':ROOT/'nr_full_evaluations/nr_conservative_s6250_compiled_all',
}
NR=['niqe','musiq','maniqa','clipiqa']
VK=['lpips','dists']+NR

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def mean(rows,split,keys):
 r=[x for x in rows if x['split']==split]
 return {k:sum(x['metrics'][k] for x in r)/len(r) for k in keys}

def nr(m):
 return m['clipiqa']+m['maniqa']+m['musiq']/100+max(0,(10-m['niqe'])/10)

def audit(label,old):
 d=RECHECK/label
 status=json.loads((d/'status.json').read_text())
 assert status['status']=='complete' and status['completed']==103 and status['scalar_metrics_completed']==418
 fresh=json.loads((d/'result.json').read_text())[label]
 prior=json.loads((old/'scores/result.json').read_text())
 p=json.loads((d/'protocol.json').read_text())
 oldp=json.loads((old/'scores/protocol.json').read_text())
 frows=[json.loads(x) for x in (d/'measurements.jsonl').read_text().splitlines() if x.strip()]
 orows=[json.loads(x) for x in (old/'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
 rp=json.loads((old/'render_protocol.json').read_text())
 manifest=json.loads((old/'render_manifest.json').read_text())
 order=[('validation',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
 assert len(frows)==len(orows)==len(manifest['records'])==len(p['expected_rows'])==103
 assert len(rp['protocol']['tasks'])==103
 assert [(x['split'],x['case']) for x in frows]==[(x['split'],x['case']) for x in orows]==order
 assert sum(len(x['metrics']) for x in frows)==sum(len(x['metrics']) for x in orows)==418
 assert p['render_protocol_id']==rp['protocol_id']
 assert p['seed']==oldp['seed']==1234
 assert p['maniqa_offline_init_proof']['test_sample']==oldp['maniqa_offline_init_proof']['test_sample']==100
 assert p['maniqa_offline_init_proof']['all_final_parameters_and_buffers_equal_checkpoint']
 for source,expected in p['frozen_source_sha256'].items():assert sha(source)==expected
 for name,rec in p['metric_weight_sha256'].items():
  path=ROOT/'scoring_tool/weights'/name
  assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
 for name,rec in json.loads((d/'artifact_hashes.json').read_text()).items():
  assert (d/name).stat().st_size==rec['bytes'] and sha(d/name)==rec['sha256']
 deltas=[]
 for f,o,expected,rec,inp in zip(frows,orows,p['expected_rows'],manifest['records'],rp['protocol']['tasks']):
  keys=VK if f['split']=='validation' else NR
  assert list(f['metrics'])==list(o['metrics'])==keys
  assert all(math.isfinite(v) for v in f['metrics'].values())
  for key in ['split','case','sr_path','sr_sha256','size_hw','input_path','input_sha256','gt_path','gt_sha256']:
   assert f[key]==expected[key]
  assert sha(f['sr_path'])==f['sr_sha256']==o['sr_sha256']==rec['sha256']
  assert list(Image.open(f['sr_path']).size)[::-1]==f['size_hw']==o['size_hw']
  assert sha(f['input_path'])==f['input_sha256']==inp['input_sha256']
  if f['split']=='validation':assert sha(f['gt_path'])==f['gt_sha256']==o['gt_sha256']
  for key in keys:
   a,b=o['metrics'][key],f['metrics'][key]
   deltas.append({'split':f['split'],'case':f['case'],'metric':key,'old':a,'fresh':b,'delta':b-a,'exact_equal':a==b})
 assert len(deltas)==418
 tm=mean(frows,'test',NR);vm=mean(frows,'validation',VK)
 N=nr(tm);F=2-vm['lpips']-vm['dists'];S=N+F
 errors=[abs(fresh['test'][k]-v) for k,v in tm.items()]+[abs(fresh['val'][k]-v) for k,v in vm.items()]
 errors += [abs(fresh['test']['nr']-N),abs(fresh['val']['fr']-F),abs(fresh['S_proxy']-S)]
 assert max(errors)<1e-12 and fresh['test']['n']==100
 per={}
 for k in VK:
  x=[v['delta'] for v in deltas if v['metric']==k]
  per[k]={'count':len(x),'exact_equal':sum(v==0 for v in x),'max_abs':max(abs(v) for v in x),
          'mean_signed':sum(x)/len(x),'mean_abs':sum(abs(v) for v in x)/len(x)}
 actual_artifacts={x.name:{'bytes':x.stat().st_size,'sha256':sha(x)} for x in d.iterdir() if x.is_file()}
 return {'label':label,'integrity_status':'PASS','image_rows':103,'scalar_metrics':418,
         'actual_PNG_manifest_old_fresh_SHA':'PASS103','actual_inputs_SHA':'PASS103','actual_GT_SHA':'PASS3',
         'fresh_sources_and_8_metric_weights_SHA':'PASS','fresh_artifact_hashes':'PASS',
         'original_seed1234_per_split_and_MANIQA100_protocol':'PASS',
         'fresh_original_score_dir_output':fresh,'fresh_CPU_recomputed':{'NR':N,'FR':F,'S_proxy':S},
         'old':{'NR':prior['test']['nr'],'FR':prior['validation']['fr'],'S_proxy':prior['S_proxy']},
         'aggregate_delta':{'NR':fresh['test']['nr']-prior['test']['nr'],'FR':fresh['val']['fr']-prior['validation']['fr'],'S_proxy':fresh['S_proxy']-prior['S_proxy']},
         'mean_then_clamp_max_error':max(errors),'per_metric_repeatability':per,
         'exact_equal_scalars':sum(x['exact_equal'] for x in deltas),'max_scalar_abs_delta':max(abs(x['delta']) for x in deltas),
         'scalar_deltas':deltas,'final_actual_artifact_sha256':actual_artifacts}

if __name__=='__main__':
 results=[audit(label,old) for label,old in PAIRS.items()]
 print(json.dumps({'audited_bjt':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  'status':'PASS','scope':'fresh original score_dir full103 /418 comparison, no new render or training',
  'CPU_only':True,'torch_imported':False,'GPU_calls':False,
  'RNG_evidence_limit':'Fresh wrapper does not save RNG snapshots; equivalence is original split seed/call protocol and observed scalar results, not a claimed byte comparison of RNG states.',
  'results':results},ensure_ascii=False,indent=2))
