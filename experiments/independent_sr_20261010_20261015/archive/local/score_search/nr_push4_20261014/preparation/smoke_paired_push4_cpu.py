"""CPU-only live planning + parentfull-source smoke; no timing data fabricated.

The existing parent is used as BOTH sides solely for the dry-run planner smoke.
Actual Push4 calibration names are checked unchanged; GPU subprocesses are blocked.
"""
from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,sys
R=Path('/cache/independent_sr_cuda_20261013');P=R/'nr_push4_20261014/preparation';P.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
oldlaunch=R/'code/push3/paired_latency_push3.py';oldjoin=R/'code/push3/join_paired_prediction_push3.py';launch=R/'code/push4/paired_latency_push4.py';join=R/'code/push4/join_paired_prediction_push4.py'
assert sha(oldlaunch)=='516bab7cb88907f04f500f8ff233e7570c0282dc49edaff775f9c240538b8ead'
assert sha(oldjoin)=='63b689536bd14bab5351efe8392115a60877d2f921778f165762129e7e3d3180'
oldsel=R/'results/nr_push3_selection.json';oldselsha=sha(oldsel);before={str(p):sha(p) for p in (R/'calibration').glob('push4_*') if p.is_file()}
l=load('push4_cpu_smoke_launcher',launch);j=load('push4_cpu_smoke_join',join)
def forbid(*a,**k):raise AssertionError('GPU/process launch forbidden during CPU smoke')
l.physical_gpu=forbid;l.subprocess.run=forbid;l.subprocess.check_output=forbid
plan=l.make_plan(oldsel,None)
assert plan['status']=='planned_not_launched' and plan['cuda_visible_devices']=='0'
assert [r['key'] for r in plan['stages']]==['g4','parent7250','selected']
assert plan['candidate']['checkpoint_sha256']==plan['parent7250']['checkpoint_sha256']=='fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e'
for stage in plan['stages']:
 assert str(stage['out']).startswith(str(R/'calibration/push4_'))
 for flag,value in [('--device','0'),('--warm','30'),('--n','200'),('--seed','231')]:assert stage['argv'][stage['argv'].index(flag)+1]==value
(P/'paired_parent_self_cpu_dryrun.json').write_text(json.dumps({'smoke_only_candidate_equals_parent':True,'no_benchmark_started':True,'plan':plan},ensure_ascii=False,indent=2)+'\n')
parent=read(oldsel);oldproof=parent['full_audit_sha256'];newproof=R/'audits/push4/evaluations/parent7250_full_cpu_audit.json';parent=dict(parent,full_audit=str(newproof),full_audit_sha256=sha(newproof));quality=j.full_quality(parent,newproof)
assert quality['image_rows']==103 and quality['scalar_metrics']==418 and quality['actual103_PNG_input_manifest_scored_SHA_verified']
assert abs(quality['NR']-2.313125129005856)<1e-12 and abs(quality['FR']-1.4465690056482952)<1e-12 and abs(quality['S']-3.7596941346541515)<1e-12
j.check_file(R/'scoring_tool/predict.py',j.PREDICT_SHA);j.check_file(R/'code/benchmark_cuda.py',j.BENCHMARK_SHA)
def functions(path):return {n.name:ast.get_source_segment(Path(path).read_text(),n) for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=functions(oldjoin),functions(join)
for name in ['full_quality','percentile','timing']:assert a[name]==b[name]
assert sha(oldsel)==oldselsha and sha(oldlaunch)=='516bab7cb88907f04f500f8ff233e7570c0282dc49edaff775f9c240538b8ead' and sha(oldjoin)=='63b689536bd14bab5351efe8392115a60877d2f921778f165762129e7e3d3180'
after={str(p):sha(p) for p in (R/'calibration').glob('push4_*') if p.is_file()};assert before==after
assert 'torch' not in sys.modules,'Smoke/join unexpectedly imported torch'
report={'status':'PASS','schema':'push4_paired_cpu_preparation_smoke_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'torch_imported':False,'GPU_subprocesses_blocked':True,'benchmark_started':False,'provisional_timing_or_prediction_written':False,'calibration_push4_files_unchanged':True,'new_launcher_sha256':sha(launch),'new_join_sha256':sha(join),'smoke_source_sha256':sha(__file__),'old_frozen_launcher_sha256':sha(oldlaunch),'old_frozen_join_sha256':sha(oldjoin),'old_frozen_selection_sha256':oldselsha,'parent_checkpoint_sha256':parent['checkpoint_sha256'],'parent_original_selection_full_audit_sha256':oldproof,'parent_push4_reaudit_sha256':sha(newproof),'parent_full103_NR':quality['NR'],'parent_full103_FR':quality['FR'],'parent_full103_S_proxy':quality['S'],'parent_actual103_PNG_input_metric_source_SHA_checked':True,'quality_percentile_timing_functions_byte_identical_to_frozen_push3':True,'planned_stage_order':['g4','parent7250','selected'],'planned_physical_gpu':0,'planned_warm':30,'planned_n':200,'planned_seed':231,'same_scope_full103_required_before_real_join':True}
(P/'paired_latency_cpu_preparation_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
