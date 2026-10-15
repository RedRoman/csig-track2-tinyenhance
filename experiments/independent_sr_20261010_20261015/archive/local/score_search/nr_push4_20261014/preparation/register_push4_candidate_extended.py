"""CPU-only, immutable registration of a new raw-checkpoint CUDA candidate.

Dry-run by default; --execute creates a fresh submission and never overwrites it.
Only checkpoint path, candidate label and explicitly requested coeff_t may differ
from the completed C7250 reference. Runner/scorer/render sources are unchanged.
"""
from pathlib import Path
import argparse, copy, datetime, hashlib, json, re, shlex

ROOT=Path('/cache/independent_sr_cuda_20261013')
REFERENCE=ROOT/'submissions/hypir_push3_c7250_compiled/model_dir'
REF_SHA='fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e'
RUNNER_SHA='396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae'
CASES=[1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100]
ALLOWED_RUNS={'cuda_nr_stage5_s7250_teacher6500_continue500','cuda_nr_stage5_s7250_teacher6500_m25_500','cuda_nr_stage5_s7250_teacher6500_m25_r10_500','cuda_nr_stage5_s7250_teacher6500_m25_r075_250'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def record(p):
    p=Path(p).resolve()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--label',required=True)
    ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--checkpoint-sha256',required=True)
    ap.add_argument('--coeff-t',type=int,default=None)
    ap.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    assert re.fullmatch(r'push4_[a-zA-Z0-9][a-zA-Z0-9_.-]{0,140}',a.label),'Use a fresh Push4 label'
    assert re.fullmatch('[0-9a-f]{64}',a.checkpoint_sha256),'Expected actual SHA256 required'
    ck=a.checkpoint.resolve(strict=True)
    assert ck.is_file() and sha(ck)==a.checkpoint_sha256,'Actual candidate checkpoint mismatch'
    assert ck.parent.parent==ROOT/'adaptation/runs' and ck.parent.name in ALLOWED_RUNS,'New stage5 run required'
    assert ck.name in {'step7500.pth','step7750.pth'},'Only completed immutable stage5 milestones7500/7750'
    if ck.parent.name=='cuda_nr_stage5_s7250_teacher6500_m25_r075_250':
        assert ck.name=='step7500.pth' and a.label=='push4_d7500_compiled','D250 only permits its actual7500 milestone and D label'
    original=json.loads((REFERENCE/'config.json').read_text())
    assert sha(original['weight_path'])==REF_SHA,'Reference checkpoint changed'
    assert sha(REFERENCE/'runner.py')==RUNNER_SHA,'Reference runner changed'
    assert original['model_t']==200 and original['coeff_t']==175
    ref_eval=ROOT/'nr_push3_full_evaluations/push3_c7250_compiled_all'
    rp=json.loads((ref_eval/'render_protocol.json').read_text())['protocol']
    sp=json.loads((ref_eval/'scores/protocol.json').read_text())
    assert sha(ROOT/'code/render_cuda_submission.py')==rp['source_sha256']
    assert sha(ROOT/'code/score_cuda_stream.py')==sp['script_sha256']
    assert sha(ROOT/'calibration/environment_protocol.json')==sp['environment_protocol_sha256']
    assert sha(ROOT/'scoring_tool/deployment_manifest.json')==sp['original_tool_deployment_sha256']
    for name,expected in rp['runner']['source_sha256'].items():
        assert sha(Path(original['repo_path'])/name)==expected
    config=copy.deepcopy(original)
    config.update(label=a.label,weight_path=str(ck))
    if a.coeff_t is not None:
        assert 1<=a.coeff_t<=1000
        config['coeff_t']=a.coeff_t
    changes={k:{'reference':original.get(k),'candidate':config.get(k)} for k in set(original)|set(config) if original.get(k)!=config.get(k)}
    assert set(changes)<=({'label','weight_path','coeff_t'} if a.coeff_t is not None else {'label','weight_path'})
    destination=ROOT/'submissions'/('hypir_'+a.label)/'model_dir'
    assert not destination.parent.exists(),'Immutable candidate already exists; use a new label'
    source_paths=[ROOT/'code/render_cuda_submission.py',ROOT/'code/score_cuda_stream.py',ROOT/'scoring_tool/deployment_manifest.json',ROOT/'calibration/environment_protocol.json',ROOT/'calibration/base_model_provenance_shared.json']
    source_paths += [Path(original['repo_path'])/name for name in ['HYPIR/enhancer/base.py','HYPIR/enhancer/sd2.py','HYPIR/utils/common.py']]
    plan={'schema':'nr_push4_candidate_registration_v1','created_bjt':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'label':a.label,'cpu_only':True,'reference_model_dir':str(REFERENCE),'reference_checkpoint_sha256':REF_SHA,'reference_runner_sha256':RUNNER_SHA,'checkpoint':record(ck),'model_dir':str(destination),'config':config,'config_differences':changes,'sources':[record(p) for p in source_paths],'registration_script_sha256':sha(__file__),'mode':'execute' if a.execute else 'dry_run','commands':{}}
    py=str(ROOT/'env/bin/python')
    for scope,folder,suffix in [('pilot','nr_push4_pilot_evaluations','pilot'),('full','nr_push4_full_evaluations','all')]:
        label=a.label+'_'+suffix;out=ROOT/folder/label
        render=[py,str(ROOT/'code/render_cuda_submission.py'),'--model-dir',str(destination),'--test-dir',str(ROOT/'source_payload/assets/input'),'--val-dir',str(ROOT/'source_payload/scoring_assets/data/验证集'),'--out-dir',str(out),'--cases','pilot' if scope=='pilot' else 'all']
        score=[py,str(ROOT/'code/score_cuda_stream.py'),'--tool-dir',str(ROOT/'scoring_tool'),'--evaluation',label+'='+str(out/'render'),'--out-root',str(ROOT/folder)]
        if scope=='pilot':score+=['--test-cases',','.join(map(str,CASES))]
        plan['commands'][scope]={'render':shlex.join(render),'score':shlex.join(score),'label':label,'scope':'S_subset only (23 images / 98 scalars)' if scope=='pilot' else 'S_proxy test100 NR + val3 FR (103 images / 418 scalars)','gpu_scheduling':'Root assigns CUDA_VISIBLE_DEVICES separately; no GPU launched by this registration.'}
    if a.execute:
        destination.mkdir(parents=True,exist_ok=False)
        (destination/'runner.py').write_bytes((REFERENCE/'runner.py').read_bytes())
        (destination/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
        assert sha(destination/'runner.py')==RUNNER_SHA
        plan['registered_files']=[record(destination/x) for x in ['runner.py','config.json']]
        (destination.parent/'registration.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
