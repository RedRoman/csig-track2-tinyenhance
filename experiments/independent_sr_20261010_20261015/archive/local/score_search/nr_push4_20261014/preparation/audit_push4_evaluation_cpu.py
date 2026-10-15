"""CPU audit of Push4 full103/418 or fixed pilot23/98 versus completed C7250."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, hashlib, json, math, datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path('/cache/independent_sr_cuda_20261013')
NR = ['niqe', 'musiq', 'maniqa', 'clipiqa']
VK = ['lpips', 'dists'] + NR
METRIC_WEIGHTS = {'clip/RN50.pt': 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762', 'torch/hub/checkpoints/DISTS_weights-f5e65c96.pth': 'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218', 'torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth': 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0', 'torch/hub/checkpoints/alexnet-owt-7be5be79.pth': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02', 'torch/hub/checkpoints/ckpt_koniq10k.pt': 'a207f8ab57322e6be38ff5c8d019301dc032b454bef21c3c9f9dbf7974eebff6', 'torch/hub/checkpoints/musiq_koniq_ckpt-e95806b9.pth': 'e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa', 'torch/hub/checkpoints/niqe_modelparameters.mat': 'fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d', 'torch/hub/checkpoints/vgg16-397923af.pth': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'}

CACHE = {}

def sha(p):
    p=Path(p).resolve()
    stamp=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
    if stamp in CACHE:return CACHE[stamp]
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    CACHE[stamp]=h.hexdigest()
    return CACHE[stamp]

def normalized(x):
    if isinstance(x, torch.Tensor):
        return {'dtype': str(x.dtype), 'shape': list(x.shape),
                'sha256': hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()}
    if isinstance(x, np.ndarray):
        return {'dtype': str(x.dtype), 'shape': list(x.shape),
                'sha256': hashlib.sha256(x.tobytes()).hexdigest()}
    if isinstance(x, dict):
        return {k: normalized(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [normalized(v) for v in x]
    return x

def rng_sha(x):
    return hashlib.sha256(json.dumps(normalized(x), sort_keys=True).encode()).hexdigest()

def nr(m):
    return m['clipiqa'] + m['maniqa'] + m['musiq']/100 + max(0, (10-m['niqe'])/10)

def audit(label, evals_root):
    d = Path(evals_root).resolve()/label
    rp = json.loads((d/'render_protocol.json').read_text())
    p = rp['protocol']
    assert hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() == rp['protocol_id']
    sp = json.loads((d/'scores/protocol.json').read_text())
    result = json.loads((d/'scores/result.json').read_text())
    status = json.loads((d/'render_status.json').read_text())
    score_status = json.loads((d/'scores/status.json').read_text())
    full=bool(result['official_full_count'])
    test_cases=list(range(1,101)) if full else [1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100]
    count=103 if full else 23
    scalars=418 if full else 98
    assert score_status['status']=='complete' and score_status['completed']==score_status['expected']==count
    manifest = json.loads((d/'render_manifest.json').read_text())
    assert manifest['protocol_id']==rp['protocol_id'] and manifest['protocol']==p
    assert sp['label']==result['label']==label
    assert Path(sp['render_dir']).resolve()==(d/'render').resolve()
    assert sp['metric_order']=={'validation':VK,'test':NR}
    rows = [json.loads(x) for x in (d/'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
    assert result['status'] == status['status'] == 'complete'
    assert result['completed'] == result['expected'] == status['completed'] == count
    assert result['official_full_count'] is full
    assert result['S_proxy_scope']==('full100_test_NR_plus3val_FR' if full else 'partial_or_preflight_only')
    assert len(rows) == len(p['tasks']) == len(manifest['records']) == count
    assert sum(len(x['metrics']) for x in rows) == scalars
    order = [('validation', i) for i in range(1,4)] + [('test', i) for i in test_cases]
    assert [(x['split'], x['case']) for x in rows] == order == [tuple(x) for x in sp['sequence']]
    records = []
    for row, rec, task in zip(rows, manifest['records'], p['tasks']):
        assert list(row['metrics']) == (VK if row['split'] == 'validation' else NR)
        assert all(math.isfinite(x) for x in row['metrics'].values())
        assert (row['split'].replace('validation','val'),row['case']) == (rec['split'],rec['case']) == (task['split'],task['case'])
        png = d/rec['output']
        assert Path(row['sr_path']).resolve()==png.resolve()
        actual = sha(png)
        assert actual == row['sr_sha256'] == rec['sha256']
        with Image.open(png) as im:
            assert im.mode=='RGB' and im.format=='PNG'
            assert list(im.size) == rec['size'] == task['size']
        assert row['size_hw'] == rec['size'][::-1]
        assert sha(task['input']) == task['input_sha256']
        gt_sha = None
        if row['split'] == 'validation':
            gt = ROOT/'scoring_tool/data/验证集'/('case3_gt.jpg' if row['case'] == 3 else f'case{row["case"]}_gt.png')
            gt_sha = sha(gt)
            assert gt_sha == row['gt_sha256']
        records.append({'split':row['split'], 'case':row['case'], 'actual_png_sha256':actual,
                        'actual_input_sha256':task['input_sha256'], 'actual_gt_sha256':gt_sha})
    means = {}
    diffs = []
    for split, keys in [('validation',VK),('test',NR)]:
        subset = [r for r in rows if r['split'] == split]
        assert result[split]['n']==len(subset)
        means[split] = {k:sum(r['metrics'][k] for r in subset)/len(subset) for k in keys}
        diffs += [abs(result[split][k]-v) for k,v in means[split].items()]
        if 'nr_val' in result[split]:
            diffs.append(abs(result[split]['nr_val']-nr(means[split])))
        if 'nr' in result[split]:
            diffs.append(abs(result[split]['nr']-nr(means[split])))
    N = nr(means['test'])
    F = 2-means['validation']['lpips']-means['validation']['dists']
    S = N+F
    diffs += [abs(result['test']['nr']-N), abs(result['validation']['fr']-F), abs(result['S_proxy']-S)]
    assert max(diffs) < 1e-12
    sources = {'renderer':sha(ROOT/'code/render_cuda_submission.py'),
               'scorer':sha(ROOT/'code/score_cuda_stream.py'),
               'runner':sha(Path(p['model_dir'])/'runner.py')}
    assert sources['renderer'] == p['source_sha256']
    assert sources['scorer'] == sp['script_sha256']
    assert sources['runner'] == p['runner']['runner_sha256']
    for name, expected in p['runner']['source_sha256'].items():
        assert sha(Path(p['runner']['repo_path'])/name) == expected
    for name, expected in p['model_dir_sha256'].items():
        assert sha(Path(p['model_dir'])/name) == expected
    for rec in sp['original_sources']:
        assert sha(rec['resolved_path']) == rec['sha256']
    assert sha(ROOT/'calibration/environment_protocol.json') == sp['environment_protocol_sha256']
    assert sha(ROOT/'scoring_tool/deployment_manifest.json') == sp['original_tool_deployment_sha256']
    assert sp['seed']==1234 and sp['maniqa_offline_init_proof']['test_sample']==100
    assert sp['maniqa_offline_init_proof']['all_final_parameters_and_buffers_equal_checkpoint']
    assert not sp['maniqa_offline_init_proof']['net_training']
    assert sp['maniqa_offline_init_proof']['checkpoint_sha256']==METRIC_WEIGHTS['torch/hub/checkpoints/ckpt_koniq10k.pt']
    metric_weights = {}
    for name, expected in METRIC_WEIGHTS.items():
        path=ROOT/'scoring_tool/weights'/name
        actual=sha(path);assert actual==expected
        metric_weights[name]={'bytes':path.stat().st_size,'sha256':actual}
    assert sha(Path.home()/'.cache/clip/RN50.pt')==METRIC_WEIGHTS['clip/RN50.pt']
    provenance=json.loads((ROOT/'calibration/base_model_provenance_shared.json').read_text())
    base=Path(p['runner']['base_model_path'])
    assert base.resolve()==Path(provenance['cuda_base_path']).resolve()
    base_expected=dict(provenance['initial_core_asset_hashes'])
    for name,record in provenance['all_config_tokenizer_scheduler_metadata_hashes'].items():
        if name in base_expected:assert base_expected[name]['sha256']==record['sha256']
        base_expected[name]=record
    base_actual={}
    for name,record in base_expected.items():
        path=base/name;actual=sha(path)
        assert actual==record['sha256'] and path.stat().st_size==record['bytes']
        base_actual[name]={'bytes':path.stat().st_size,'sha256':actual}
    for key, filekey, shakey in [('numpy_legacy_compat','path','sha256'),('maniqa_offline_init_proof','adapter_path','adapter_sha256')]:
        assert sha(sp[key][filekey]) == sp[key][shakey]
    checkpoint = sha(p['runner']['weight_path'])
    assert checkpoint == p['runner']['checkpoint_sha256']
    render_state = torch.load(d/'render_progress.pt', map_location='cpu', weights_only=False)
    score_state = torch.load(d/'scores/rng_checkpoint.pt', map_location='cpu', weights_only=False)
    assert render_state['protocol_id'] == rp['protocol_id']
    assert render_state['records'] == manifest['records']
    assert render_state['checkpoint_sha256']==checkpoint
    assert render_state['input_sha256']==[task['input_sha256'] for task in p['tasks']]
    invariant=json.loads(json.dumps(sp))
    for call in invariant['maniqa_offline_init_proof']['before_init_records']:
        call.pop('before_original_checkpoint_load',None)
    assert score_state['fingerprint']==hashlib.sha256(json.dumps(invariant,sort_keys=True).encode()).hexdigest()
    assert score_state['rows'] == rows and score_state['active_split'] == 'test'
    assert not torch.cuda.is_initialized()
    return {'label':label,'audited_bjt':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'scope':'full test100 + validation3 local proxy; no latency/official total' if full else 'fixed20 test + validation3 pilot; S_subset only; not full score',
            'cpu_only':True,'cuda_initialized':False,'status':'PASS', 'image_rows':count,'scalar_metrics':scalars,
            'test_cases':test_cases,'official_full_count':full,'NR':N,'FR':F,('S_proxy' if full else 'S_subset'):S,'component_means':means,
            'mean_formula_max_delta':max(diffs),'actual_png_manifest_scored_sha':'PASS'+str(count),
            'actual_source_input_sha':'PASS'+str(count),'actual_source_GT_sha':'PASS3','actual_source_code_sha':'PASS',
            'actual_checkpoint_sha256':checkpoint,'current_source_sha256':sources,
            'actual_metric_weights_sha256':metric_weights,'actual_base_assets_sha256':base_actual,
            'published_result':result,'evaluation_dir':str(d),'score_context_fingerprint_matches_protocol':True,
            'render_protocol_id':rp['protocol_id'],'render_seed':p['runner']['seed'],'score_seed':sp['seed'],
            'score_seed_scope':sp['seed_scope'],'metric_order':sp['metric_order'],
            'render_rng_policy':p['inference_rng'],'score_rng_policy':sp['resume_policy'],
            'render_post_rng_hashes':{k:rng_sha(render_state[k]) for k in ['python_rng','numpy_rng','torch_rng','device_rng']},
            'score_post_rng_hashes':{k:rng_sha(v) for k,v in score_state['rng'].items()},
            'render_state_commits_equal_manifest':True,'score_state_rows_equal_jsonl':True,
            'records':records}



def compare_reference(candidate_dir, reference_dir, candidate_audit, reference_audit):
    candidate_dir,reference_dir=Path(candidate_dir),Path(reference_dir)
    cp=json.loads((candidate_dir/'render_protocol.json').read_text())['protocol']
    rp=json.loads((reference_dir/'render_protocol.json').read_text())['protocol']
    cs=json.loads((candidate_dir/'scores/protocol.json').read_text())
    rs=json.loads((reference_dir/'scores/protocol.json').read_text())
    assert candidate_audit['test_cases']==reference_audit['test_cases']
    assert cp['tasks']==rp['tasks'],'Inputs/order/native shape changed'
    for k in ['schema','cases','input_encoding','output_encoding','warmup','inference_rng','source_sha256','torch']:
        assert cp[k]==rp[k],k
    runner_differences={k:{'reference':rp['runner'].get(k),'candidate':cp['runner'].get(k)} for k in set(cp['runner'])|set(rp['runner']) if cp['runner'].get(k)!=rp['runner'].get(k)}
    assert set(runner_differences)<={'label','weight_path','checkpoint_sha256','runner_path','coeff_t'},runner_differences
    assert cp['runner']['model_t']==200
    formal_keys=['metric_source','original_tool_deployment_sha256','original_sources','script_sha256','environment_protocol_sha256','environment_versions','numpy_legacy_compat','seed','seed_scope','metric_order','image_policy','resume_policy','interleaving_policy','sequence']
    for k in formal_keys:assert cs[k]==rs[k],k
    # Untrained-init evidence is random; final loaded checkpoint proof must match.
    cproof=json.loads(json.dumps(cs['maniqa_offline_init_proof']))
    rproof=json.loads(json.dumps(rs['maniqa_offline_init_proof']))
    for proof in [cproof,rproof]:
        for row in proof['before_init_records']:row.pop('before_original_checkpoint_load',None)
    assert cproof==rproof,'Formal MANIQA final checkpoint/crops changed'
    render_rng_equal=candidate_audit['render_post_rng_hashes']==reference_audit['render_post_rng_hashes']
    score_rng_equal=candidate_audit['score_post_rng_hashes']==reference_audit['score_post_rng_hashes']
    assert render_rng_equal,'Ordered native inference RNG draws changed'
    assert score_rng_equal,'Ordered formal metric RNG draws changed'
    field='S_proxy' if candidate_audit['official_full_count'] else 'S_subset'
    return {'reference_dir':str(reference_dir.resolve()),'scope':candidate_audit['scope'],'same_original_inputs_GT_and_order':True,'same_formal_metrics_and_sources':True,'same_render_input_output_and_rng_policy':True,'same_post_render_rng':render_rng_equal,'same_post_score_rng':score_rng_equal,'runner_differences':runner_differences,'reference_checkpoint_sha256':reference_audit['actual_checkpoint_sha256'],'candidate_checkpoint_sha256':candidate_audit['actual_checkpoint_sha256'],'delta_NR':candidate_audit['NR']-reference_audit['NR'],'delta_FR':candidate_audit['FR']-reference_audit['FR'],'delta_'+field:candidate_audit[field]-reference_audit[field]}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--evals-root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=None,help='Completed C7250 evaluation in identical pilot/full scope')
    args=parser.parse_args()
    candidate=audit(args.label,args.evals_root)
    if args.reference is None:
        folder='nr_push3_full_evaluations' if candidate['official_full_count'] else 'nr_push3_pilot_evaluations'
        suffix='all' if candidate['official_full_count'] else 'pilot'
        args.reference=ROOT/folder/('push3_c7250_compiled_'+suffix)
    reference=audit(args.reference.name,args.reference.parent)
    assert reference['actual_checkpoint_sha256']=='fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e','Reference must be sealed C7250 checkpoint'
    candidate['reference_comparison']=compare_reference(args.evals_root/args.label,args.reference,candidate,reference)
    candidate['auditor_source_sha256']=sha(__file__)
    assert not torch.cuda.is_initialized()
    print(json.dumps(candidate,ensure_ascii=False,indent=2))
