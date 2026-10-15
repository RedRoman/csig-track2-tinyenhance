"""Push7 generic CPU full103/418 audit and fresh-triplet join; default plan.

No Runner imports, GPU work, metric changes, or candidate quality reuse.
--mode audit reads the selected actual formal render and fresh score418.
--mode join consumes that independently recorded audit, unchanged-file guards,
new same-GPU0 G4/Push6-parent/selected raw200, and root launch evidence.
"""
import argparse
from datetime import datetime, timezone
import datetime as datetime_module
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys

sys.dont_write_bytecode = True
R = ROOT = Path('/cache/independent_sr_cuda_20261013')
RAW = R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
RAW_SHA = '589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
BENCH_SHA = '3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
PRED_SHA = '0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
RENDER_SHA = '1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
SCORER_SHA = '20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074'
CM_SHA = '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
PARENT = R/'nr_push4_full_evaluations/push4_a7750_compiled_all'
PARENT_MODEL = R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
PARENT_RUNNER = 'dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
PARENT_CONFIG = '8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8'
PARENT_AUDIT = R/'audits/push4/evaluations/push4_a7750_compiled_all/audit.json'
PARENT_AUDIT_SHA = 'ae5b2234f5e110240507fbec515139d1b2a23bf8f664e7477e34c377e09c507b'
PUSH6_EXACT = R/'audits/push6/formal_png103/ro_wavelet_exact103_v2.json'
PUSH6_EXACT_SHA = 'bc5dff12c258069bb77e5f3e0b6982737a9d74ea705782bb230ef244b1850e16'
ORDER = [('val',i) for i in range(1,4)] + [('test',i) for i in range(1,101)]
NR = ['niqe','musiq','maniqa','clipiqa']
VK = ['lpips','dists'] + NR
SEEN = {}
MODELS = {}

METRIC_WEIGHTS = {'clip/RN50.pt': 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762', 'torch/hub/checkpoints/DISTS_weights-f5e65c96.pth': 'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218', 'torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth': 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0', 'torch/hub/checkpoints/alexnet-owt-7be5be79.pth': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02', 'torch/hub/checkpoints/ckpt_koniq10k.pt': 'a207f8ab57322e6be38ff5c8d019301dc032b454bef21c3c9f9dbf7974eebff6', 'torch/hub/checkpoints/musiq_koniq_ckpt-e95806b9.pth': 'e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa', 'torch/hub/checkpoints/niqe_modelparameters.mat': 'fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d', 'torch/hub/checkpoints/vgg16-397923af.pth': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'}

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

def audit_core(label, evals_root):
    from PIL import Image
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
    return {'label':label,'audited_bjt':datetime_module.datetime.now(datetime_module.timezone(datetime_module.timedelta(hours=8))).isoformat(),
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

def need(ok, message):
    if not ok: raise RuntimeError(message)

def token(p):
    s = Path(p).stat()
    return s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev

def sha(p):
    p = Path(p).resolve(); before = token(p)
    if p in SEEN:
        need(SEEN[p][0] == before, f'Source changed: {p}')
        return SEEN[p][1]
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    need(token(p) == before, f'Source changed during SHA: {p}')
    SEEN[p] = before, h.hexdigest()
    return h.hexdigest()

def read(p):
    sha(p)
    return json.loads(Path(p).read_text())

def ref(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p), bytes=Path(p).stat().st_size)

def checked(p, digest):
    need(sha(p) == digest, f'Actual SHA differs: {p}')

def percentile(values, fraction):
    x = sorted(values); pos = (len(x) - 1) * fraction; lo = int(pos)
    return x[lo] + (x[min(lo + 1, len(x) - 1)] - x[lo]) * (pos - lo)

def timing(key, directory):
    name, model, runner_sha = MODELS[key]; path = directory / name; x = read(path)
    need(x['status'] == 'complete' and Path(x['model_dir']).resolve() == model, 'Timing status/model differs')
    need(x['seed'] == 231 and x['latency_512']['n'] == 200 and x['latency_512']['warm'] == 30, 'Timing protocol differs')
    need(x['input']['shape'] == [1,3,512,512] and x['input']['dtype'] == 'torch.float16'
         and x['input']['device'] == 'cuda:0' and x['cuda']['logical_device'] == 0
         and x['input']['distribution']=='uniform [-1,1)' and x['input']['allocation_excluded_from_timing'] is True,
         'Timing CUDA input differs')
    y = x['output']
    need(y['device'] == 'cuda:0' and y['shape'] == [1,3,512,512] and y['all_finite'] is True
         and y['within_minus1_plus1'] is True and -1 <= y['min'] <= y['max'] <= 1
         and y['clamping_applied_by_benchmark'] is False, 'Timing output invalid')
    need(y['dtype']==('torch.float16' if key=='g4' else 'torch.float32'),'Wrong complete-infer output dtype')
    need(x['benchmark_script']['sha256'] == BENCH_SHA, 'Wrong benchmark source')
    checked(R/'code/benchmark_cuda.py', BENCH_SHA)
    records = x['artifacts']['files'] + x['artifacts']['extra_files']
    need(not x['artifacts']['untraversed_symlink_directories'], 'Unbound symlink directory')
    for row in records:
        p = Path(row['resolved_path']); checked(p, row['sha256'])
        need(p.stat().st_size == row['bytes'], 'Artifact size changed')
    checked(model/'runner.py', runner_sha)
    need(any(Path(z['resolved_path']) == model/'runner.py' and z['sha256'] == runner_sha for z in records), 'Runner not bound')
    if key != 'g4':
        need(any(Path(z['resolved_path']) == RAW and z['sha256'] == RAW_SHA for z in records), 'A7750 raw not bound')
    else:
        need(any(z['sha256'] == '283cb7d051a4686e558057b8fb0dcd927b47a8cb7474101fab4162c457c2bb83' for z in records), 'Wrong G4 reference')
    computed = {}
    for kind in ['event', 'wall']:
        v = x['raw_samples'][kind+'_ms']
        need(len(v) == 200 and all(isinstance(t,(int,float)) and math.isfinite(t) and t > 0 for t in v), 'Invalid raw200')
        for suffix, number in [('p10',percentile(v,.1)), ('med',statistics.median(v)), ('min',min(v))]:
            k = kind+'_'+suffix
            need(abs(number-x['latency_512'][k]) <= 1e-9, 'Timing raw summary mismatch: '+k)
            computed[k] = number
    return dict(artifact=ref(path), model_dir=str(model), runner_sha256=runner_sha,
                seed=231, n=200, warm=30, input=x['input'], output=y, cuda=x['cuda'],
                recomputed=computed, actual_artifact_SHA_verified=True, raw200_verified=True)

def launch(directory):
    p = directory/'root_launch_evidence.json'; x = read(p)
    need(x['schema'] == 'push7_root_launch_evidence_v1' and x['serial'] is True
         and x['order'] == ['g4','parent','candidate'] and x['other_root_gpu_jobs_active_during_pair'] is False, 'Root launch scope differs')
    need([z['label'] for z in x['commands']] == x['order'], 'Command order differs')
    for i, row in enumerate(x['commands'], 1):
        need(row['order'] == i and row['exit_code'] == 0 and row['separate_fresh_process'] is True, 'Launch exit/fresh proof missing')
        argv = shlex.split(row['command']); name, model, _ = MODELS[row['label']]
        need(argv[0] == 'CUDA_VISIBLE_DEVICES=0' and Path(argv[1]) == R/'env/bin/python'
             and Path(argv[2]) == R/'code/benchmark_cuda.py', 'Launch command source/device differs')
        for flag, expected in [('--model-dir',str(model)),('--out',str(directory/name)),('--device','0'),('--seed','231'),('--warm','30'),('--n','200')]:
            need(flag in argv and argv[argv.index(flag)+1] == expected, 'Launch flag differs: '+flag)
    files = [directory/n for n in ['gpu_before_candidate.csv','gpu_after_candidate.csv','processes_before_candidate.csv','processes_after_candidate.csv']]
    gpu = [f.read_text().strip() for f in files[:2]]
    need(gpu[0] == gpu[1] and len(gpu[0].split(',')) == 4, 'Physical candidate snapshots differ')
    need(all(not f.read_text().strip() for f in files[2:]), 'Candidate snapshot showed another process')
    fields = [s.strip() for s in gpu[0].split(',')]
    need(fields[0] == '0' and fields[1].startswith('GPU-'), 'Bad physical snapshot')
    for f in files: sha(f)
    return dict(root_evidence=ref(p), physical_slot=0, UUID=fields[1], PCI=fields[2],
                snapshots=[ref(f) for f in files], order=x['order'], all_exit0=True,
                evidence_scope='Root actual fresh CUDA_VISIBLE_DEVICES0 command/exit records plus candidate-only physical before/after snapshots; no before-G4 snapshot claimed')


def stable_files():
    for p, (stamp, digest) in SEEN.items():
        need(token(p) == stamp, 'Read source changed before publish: '+str(p))


def binding(model):
    mp = model.parent/'candidate_manifest.json'; m = read(mp); c = read(model/'config.json')
    need(Path(m['model_dir']).resolve() == model and m['label'] == c['label'], 'Manifest/model/label mismatch')
    checked(model/'runner.py', m['runner_sha256']); checked(model/'config.json', m['config_sha256'])
    need(m['checkpoint_sha256'] == RAW_SHA and Path(c['weight_path']).resolve() == RAW and
         Path(m['checkpoint']).resolve() == RAW, 'Not immutable A7750 raw')
    need(m.get('quality_reuse_claim') is False, 'Push7 quality must be newly scored')
    need(m.get('channels_last_conversion') is False, 'Unexpected channels-last branch')
    need(c['seed']==231 and c['model_t']==200 and c['coeff_t']==175 and c['lora_rank']==256 and
         c['internal_precision']=='bfloat16' and c['upscale']==1 and c['caption']=='', 'Changed HYPIR core protocol')
    need(c['compute_scale'] in [.75,.625,.5] and
         (c['patch_size'],c['stride']) == {.75:(384,192),.625:(320,160),.5:(256,128)}[c['compute_scale']] and
         c['reconstruction_mode'] in ['residual','direct'], 'Undeclared compute/reconstruction policy')
    return m,c,ref(mp)


def metadata_paths(d):
    return [d/n for n in ['render_status.json','render_protocol.json','render_manifest.json','render_progress.pt',
            'scores/status.json','scores/protocol.json','scores/result.json','scores/measurements.jsonl','scores/rng_checkpoint.pt']]


def perform_audit(d, model):
    global torch,np
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import numpy as np
    from PIL import Image
    need(not torch.cuda.is_initialized(), 'CPU audit must not initialize CUDA')
    need(d.is_relative_to(R/'nr_push7_full_evaluations'), 'Selected quality must be a new Push7 full evaluation')
    need((d/'render_complete').is_file(), 'Full render completion marker missing')
    m,c,mref = binding(model)
    for p in metadata_paths(d): sha(p)
    published=read(d/'scores/result.json')
    need(published['official_full_count'] is True and published['completed']==published['expected']==103,
         'Fresh full103/418 only; never a pilot result')
    rp=read(d/'render_protocol.json');p=rp['protocol'];rr=p['runner']
    need(Path(p['model_dir']).resolve()==model and rr['runner_sha256']==m['runner_sha256'] and
         p['model_dir_sha256']['config.json']==m['config_sha256'] and rr['label']==c['label'], 'Formal render is not manifest-selected runtime')
    need(p['source_sha256']==RENDER_SHA and p['schema']=='cuda_submission_quality_v1' and p['cases']=='all' and
         p['warmup'] is False and p['inference_rng']=='seed231 before model initialization; continuous ordered val3 then tests', 'Formal render policy changed')
    need(rr['seed']=='231 before initialization; continuous inference RNG' and rr['model_t']==200 and rr['coeff_t']==175 and
         rr['lora_rank']==256 and rr['internal_precision']=='BF16' and rr['scale']==1 and rr['caption']=='', 'Formal HYPIR core settings changed')
    need(rr['compute_scale']==c['compute_scale'] and rr['patch']==c['patch_size'] and rr['stride']==c['stride'] and
         rr['reconstruction_mode']==c['reconstruction_mode'] and rr['quality_reuse_allowed'] is False and
         rr['changed_pixels_and_random_consumption'] is True, 'Scaled runtime parameters/provenance differ')
    checked(R/'code/render_cuda_submission.py',RENDER_SHA);checked(R/'code/score_cuda_stream.py',SCORER_SHA)
    checked(R/'scoring_tool/csig_metrics.py',CM_SHA);checked(RAW,RAW_SHA)
    manifest=read(d/'render_manifest.json');tasks=p['tasks'];records=manifest['records']
    need(len(tasks)==len(records)==103, 'Formal tasks/records must be103')
    for index,((split,case),task,row) in enumerate(zip(ORDER,tasks,records)):
        src=R/'source_payload/scoring_assets/data/验证集'/f'case{case}_lq.jpg' if split=='val' else R/'source_payload/assets/input'/f'case{case}.jpg'
        relative=f'render/{split}/case{case}.png'
        need((task['split'],task['case'])==(split,case) and Path(task['input']).resolve()==src.resolve(), 'Not canonical formal source/order')
        need((row['task_index'],row['split'],row['case'],row['output'])==(index,split,case,relative), 'Noncanonical render index/path/order')
        need(not (d/relative).is_symlink(), 'Formal PNG must be actual file, not symlink')
        with Image.open(src) as image: need(list(image.size)==task['size'], 'Input native size changed')
    need({str(f.relative_to(d)) for f in (d/'render').rglob('*') if f.is_file()}=={r['output'] for r in records}, 'Extra/missing formal PNGs')
    core=audit_core(d.name,d.parent)
    need(core['image_rows']==103 and core['scalar_metrics']==418 and core['official_full_count'] is True,
         'Original metrics core did not audit full103/418')
    rows=[json.loads(line) for line in (d/'scores/measurements.jsonl').read_text().splitlines() if line.strip()]
    for row in rows:
        need(row['seed_scope']=='split_seed1234_continuous_rng', 'Per-image score seed metadata changed')
        need(abs(nr(row['metrics'])-row['nr_per_image'])<=1e-12, 'Per-image original NR/clamp formula differs')
        if row['split']=='test': need(row['gt_sha256'] is None,'Test metrics may not use GT')
    sp=read(d/'scores/protocol.json')
    need(sp['script_sha256']==SCORER_SHA and sp['metric_order']=={'validation':VK,'test':NR} and sp['seed']==1234,
         'Wrong original metric protocol/order/seed')
    # Metric draw shapes stay native and metric-source order remains fixed, even
    # though the scaled renderer consumes a different number of latent samples.
    checked(PARENT_AUDIT,PARENT_AUDIT_SHA);parent=read(PARENT_AUDIT)
    for task,record,frozen in zip(tasks,core['records'],parent['records']):
        need((record['split'],record['case'])==(frozen['split'],frozen['case']) and
             task['input_sha256']==frozen['actual_input_sha256'] and
             record['actual_gt_sha256']==frozen['actual_gt_sha256'],
             'Current formal input/GT differs from the frozen A comparison dataset')
    parent_render=read(PARENT/'render_protocol.json')['protocol']
    for k in ['schema','cases','input_encoding','output_encoding','warmup','inference_rng','source_sha256','torch']:
        need(p[k]==parent_render[k], 'Formal renderer encoding/environment differs: '+k)
    need(core['score_post_rng_hashes']==parent['score_post_rng_hashes'], 'Formal full metric random draw sequence differs from frozen A')
    parent_sp=read(PARENT/'scores/protocol.json')
    formal=['metric_source','original_tool_deployment_sha256','original_sources','script_sha256','environment_protocol_sha256',
            'environment_versions','numpy_legacy_compat','seed','seed_scope','metric_order','image_policy','resume_policy','interleaving_policy','sequence']
    for k in formal: need(sp[k]==parent_sp[k],'Formal scorer setting changed: '+k)
    proofs=[json.loads(json.dumps(q['maniqa_offline_init_proof'])) for q in [sp,parent_sp]]
    for proof in proofs:
        for row in proof['before_init_records']:row.pop('before_original_checkpoint_load',None)
    need(proofs[0]==proofs[1], 'Formal MANIQA final weights/crops/eval settings changed')
    core.update(schema='push7_fresh_full_quality_audit_v1',status='PASS',quality_mode='fresh_original103_418',quality_reused=False,
        model_dir=str(model),candidate_manifest=mref,config=ref(model/'config.json'),runner=ref(model/'runner.py'),
        reconstruction_mode=c['reconstruction_mode'],compute_scale=c['compute_scale'],patch_size=c['patch_size'],stride=c['stride'],
        renderer_source=ref(R/'code/render_cuda_submission.py'),scorer_source=ref(R/'code/score_cuda_stream.py'),
        candidate_render_manifest=ref(d/'render_manifest.json'),candidate_score_records=ref(d/'scores/measurements.jsonl'),
        audit_source=ref(__file__),auditor_source_sha256=sha(__file__),metric_rng_matches_frozen_full_A=True,
        render_rng_equality_to_A_required=False,render_rng_scope='actual final committed4RNG, scaling is allowed to change inference draw shapes',
        actual_render_file_count=103,extra_render_files=0,no_latency_claim=True,no_official_prediction=True)
    stable_files()
    core['audited_files']=[dict(path=str(p),bytes=stamp[0],mtime_ns=stamp[1],inode=stamp[2],device=stamp[3],sha256=digest)
                           for p,(stamp,digest) in SEEN.items()]
    return core


def consume_quality(path,d,model):
    if not path.is_file(): return None
    q=read(path)
    if q.get('status')!='PASS': return None
    need(q.get('schema')=='push7_fresh_full_quality_audit_v1' and q.get('quality_mode')=='fresh_original103_418' and
         q.get('quality_reused') is False and q['image_rows']==103 and q['scalar_metrics']==418, 'Fresh selected103/418 audit required')
    need(Path(q['evaluation_dir']).resolve()==d and Path(q['model_dir']).resolve()==model and
         q['actual_checkpoint_sha256']==RAW_SHA, 'Quality audit belongs to another runtime/evaluation')
    need(q['audit_source']['sha256']==sha(__file__), 'Quality audit produced by another helper source')
    for row in q['audited_files']:
        need(token(row['path'])==(row['bytes'],row['mtime_ns'],row['inode'],row['device']), 'Previously audited quality input changed: '+row['path'])
    m,c,mref=binding(model)
    need(q['candidate_manifest']==mref and q['runner']['sha256']==m['runner_sha256'] and q['config']['sha256']==m['config_sha256'], 'Audited candidate manifest/config changed')
    published=read(d/'scores/result.json')
    need(published==q['published_result'], 'New-score result changed after independent audit')
    rows_path=d/'scores/measurements.jsonl';checked(rows_path,q['candidate_score_records']['sha256'])
    rows=[json.loads(s) for s in rows_path.read_text().splitlines() if s.strip()]
    need(len(rows)==103 and sum(len(r['metrics']) for r in rows)==418,'Not actual fresh418 scalar records')
    means={s:{k:statistics.fmean(r['metrics'][k] for r in rows if r['split']==s) for k in (VK if s=='validation' else NR)} for s in ['validation','test']}
    n=nr(means['test']);f=2-means['validation']['lpips']-means['validation']['dists']
    need(all(abs(v-q[k])<=1e-12 for k,v in [('NR',n),('FR',f),('S_proxy',n+f)]),'Fresh quality formula differs at join')
    return dict(NR=n,FR=f,S_proxy=n+f,independent_audit=ref(path),quality_mode='fresh_original103_418',
                quality_reused=False,audited_file_stats_unchanged=True,actual_PNG_rehashed_in_this_join=False,
                score_result=ref(d/'scores/result.json'),score_records=ref(rows_path),candidate_manifest=mref)


def parent_quality():
    checked(PARENT_AUDIT,PARENT_AUDIT_SHA);q=read(PARENT_AUDIT)
    checked(PUSH6_EXACT,PUSH6_EXACT_SHA);bridge=read(PUSH6_EXACT)
    need(bridge['status']=='PASS_EXACT103' and bridge['PNG_exact_count']==103 and bridge['final_fourRNG_equal'] and
         bridge['candidate_runner_sha256']==PARENT_RUNNER and bridge['metric_input_reuse_eligible'],'Push6 parent quality bridge invalid')
    checked(PARENT_MODEL/'runner.py',PARENT_RUNNER);checked(PARENT_MODEL/'config.json',PARENT_CONFIG)
    need(read(PARENT/'scores/result.json')==q['published_result'],'Frozen parent result changed')
    return dict(NR=q['NR'],FR=q['FR'],S_proxy=q['S_proxy'],source='frozen independent A full418 plus Push6 actual exact103 bridge',
                independent_full_audit=ref(PARENT_AUDIT),Push6_exact103_bridge=ref(PUSH6_EXACT))


def publish(path,value):
    need(not path.exists(),'Use a new immutable output label; refusing to overwrite '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    global MODELS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['audit','join'],default='audit')
    p.add_argument('--evaluation-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--quality-audit',type=Path);p.add_argument('--timing-dir',type=Path)
    a=p.parse_args()
    d=a.evaluation_dir.resolve();model=a.model_dir.resolve();out=a.out.resolve()
    need(out.is_relative_to(R/'audits/push7') or out.is_relative_to(R/'calibration/push7_final'), 'New Push7 output scope required')
    need(not out.exists(),'Output already exists; use an independent new label')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',mode=a.mode,evaluation_dir=str(d),model_dir=str(model),out=str(out),
            full_image_count=103,scalar_metrics=418,render_seed=231,score_split_seed=1234,GPU=False,
            quality_policy='candidate fresh original metrics418 required, never A score reuse',
            paired_plan='fresh idleGPU0 G4→Push6-parent→selected; original benchmark warm30/n200/seed231; filenames g4_latency.json,parent_latency.json,candidate_latency.json; root_launch_evidence schema push7_root_launch_evidence_v1')))
        return
    if a.mode=='audit':
        result=perform_audit(d,model)
    else:
        need(a.timing_dir is not None and a.quality_audit is not None, 'Join requires timing-dir and quality-audit')
        m,c,mref=binding(model)
        MODELS={'g4':('g4_latency.json',R/'calibration/g4/model_dir','6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'),
                'parent':('parent_latency.json',PARENT_MODEL,PARENT_RUNNER),
                'candidate':('candidate_latency.json',model,m['runner_sha256'])}
        tdir=a.timing_dir.resolve();need(tdir.is_relative_to(R/'calibration/push7_final'),'Fresh timing must be new final scope, never exploratory')
        ts={k:timing(k,tdir) for k in MODELS};physical=launch(tdir)
        need(all(t['cuda']==ts['g4']['cuda'] for t in ts.values()),'Triplet CUDA environment differs')
        result=dict(schema='push7_fresh_paired_prediction_v1',status='PENDING_QUALITY',CPU_only=True,utc=datetime.now(timezone.utc).isoformat(),
            timings=ts,launch=physical,candidate_manifest=mref,source=ref(__file__),candidate_quality_confirmed=False,
            candidate_prediction_available=False,quality_audit_requested=str(a.quality_audit.resolve()),
            speed_only=dict(parent_event_p10_ms=ts['parent']['recomputed']['event_p10'],candidate_event_p10_ms=ts['candidate']['recomputed']['event_p10']))
        selected=consume_quality(a.quality_audit.resolve(),d,model)
        if selected:
            parent=parent_quality();predpath=R/'scoring_tool/predict.py';checked(predpath,PRED_SHA)
            spec=importlib.util.spec_from_file_location('original_predict',predpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            predictions={};r=ts['g4']['recomputed']['event_p10']
            for name,q in [('parent',parent),('candidate',selected)]:
                ms=ts[name]['recomputed']['event_p10'];g=ms*.431/r
                predictions[name]=dict(NR=q['NR'],FR=q['FR'],S_proxy=q['S_proxy'],event_p10_ms=ms,normalized_g_ms=g,original_tool_prediction=module.predict(q['S_proxy'],g))
            result.update(status='PASS',quality=dict(parent=parent,candidate=selected),predictions=predictions,
                candidate_quality_confirmed=True,candidate_prediction_available=True,target=65,reaches_target=predictions['candidate']['original_tool_prediction']>=65,
                predict_source=ref(predpath),score_scope='supplied empirical predictor only; not official contest result',
                latency_reduction_fraction=1-ts['candidate']['recomputed']['event_p10']/ts['parent']['recomputed']['event_p10'])
        stable_files()
    publish(out,result)
    print(json.dumps(dict(status=result['status'],out=str(out),sha256=sha(out),mode=a.mode)))


if __name__=='__main__': main()
