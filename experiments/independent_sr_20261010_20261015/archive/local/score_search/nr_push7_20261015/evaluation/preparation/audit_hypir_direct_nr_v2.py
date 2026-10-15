"""Minimal CPU audit for directly fine-tuned HYPIR LoRA; default plan.

New candidate raw SHA is distinct from initial A7750 provenance. The evaluated
native512 Push6 runner source is unchanged. Actual new original metrics103/418
are required; no teacher outputs, distilled model or inherited candidate score.
"""
import argparse
from datetime import datetime, timezone
import datetime as datetime_module
import hashlib,json,math,os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
R=ROOT=Path('/cache/independent_sr_cuda_20261013')
INITIAL_A=R/'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
INITIAL_A_SHA='589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
RUNNER_SHA='dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b'
RENDER_SHA='1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
SCORER_SHA='20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074'
CM_SHA='1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
PARENT=R/'nr_push4_full_evaluations/push4_a7750_compiled_all'
PARENT_MODEL=R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir'
PARENT_AUDIT=R/'audits/push4/evaluations/push4_a7750_compiled_all/audit.json'
PARENT_AUDIT_SHA='ae5b2234f5e110240507fbec515139d1b2a23bf8f664e7477e34c377e09c507b'
ORDER=[('val',i) for i in range(1,4)]+[('test',i) for i in range(1,101)]
NR=['niqe','musiq','maniqa','clipiqa'];VK=['lpips','dists']+NR
SEEN={}
METRIC_WEIGHTS = {'clip/RN50.pt': 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762', 'torch/hub/checkpoints/DISTS_weights-f5e65c96.pth': 'f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218', 'torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth': 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0', 'torch/hub/checkpoints/alexnet-owt-7be5be79.pth': '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02', 'torch/hub/checkpoints/ckpt_koniq10k.pt': 'a207f8ab57322e6be38ff5c8d019301dc032b454bef21c3c9f9dbf7974eebff6', 'torch/hub/checkpoints/musiq_koniq_ckpt-e95806b9.pth': 'e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa', 'torch/hub/checkpoints/niqe_modelparameters.mat': 'fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d', 'torch/hub/checkpoints/vgg16-397923af.pth': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'}

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

def stable_files():
    for p, (stamp, digest) in SEEN.items():
        need(token(p) == stamp, 'Read source changed before publish: '+str(p))

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


def document(record,what):
    need(isinstance(record,dict) and set(record)>={'path','sha256'}, 'Missing provenance reference: '+what)
    p=Path(record['path']).resolve();checked(p,record['sha256']);return p


def training_binding(manifest,candidate_sha):
    t=manifest['training_provenance']
    config_path=document(t['config'],'actual training config');config=read(config_path);p=config['protocol']
    need(config['protocol_id']==hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(), 'Actual training protocol digest mismatch')
    marker_path=document(t['marker'],'actual immutable checkpoint marker');marker=read(marker_path)
    need(marker['protocol_id']==config['protocol_id'] and marker['raw_checkpoint_sha256']==candidate_sha,
         'Actual raw/config/marker identity mismatch')
    local=marker['local_step'];global_step=marker['global_step']
    need(isinstance(local,int) and local in [50,100] and marker['optimizer_step']==local and global_step==7750+local,
         'Only actual local50/100 fresh-Adam HYPIR milestones supported')
    need(marker['initial_A_checkpoint_sha256']==INITIAL_A_SHA and marker['fresh_Adam_initial_step']==0 and
         marker['no_teacher_outputs'] is True and marker['no_val_GT'] is True and marker['changed_FP32_LoRA_tensors']>0,
         'Actual marker is not changed directNR/no teacher/noGT/freshAdam0 output')
    need(p['parent_raw_sha256']==INITIAL_A_SHA and p['new_Adam_initial_step']==0 and p['teacher_output_loss'] is False and
         p['distillation'] is False and p['validation_GT_training'] is False and p['stage_start_global_step']==7750 and p['stage_steps']==100,
         'Actual training config lineage/data policy changed')
    need(p['objective']['lr']==1e-6 and p['source_sha256']==marker['source_sha256'],'Training source/LR binding changed')
    for path,digest in p['source_sha256'].items():checked(path,digest)
    if 'sources' in t:
        need({x['path']:x['sha256'] for x in t['sources']}==p['source_sha256'],'Manifest training sources differ')
    inputs_path=document(t['input_catalog'],'actual input catalog')
    need(inputs_path==Path(marker['input_catalog']).resolve() and sha(inputs_path)==marker['input_catalog_sha256'], 'Marker input catalog mismatch')
    inputs=read(inputs_path)
    need(inputs==p['inputs'] and isinstance(inputs,list) and len(inputs)==100 and [x['case'] for x in inputs]==list(range(1,101)),
         'DirectNR uses exactly declared100 original test inputs')
    frozen=read(PARENT_AUDIT)['records'][3:]
    for row,old in zip(inputs,frozen):
        src=R/'source_payload/assets/input'/f'case{row["case"]}.jpg'
        need(Path(row['path']).resolve()==src.resolve() and row['sha256']==old['actual_input_sha256'], 'Training input differs from frozen original test cohort')
        need(not({'teacher','teacher_sha256','target','gt','gt_sha256'} & set(row)), 'Teacher/GT data in directNR input catalog')
        checked(src,row['sha256'])
    resume=document(t['resume'],'actual immutable matching resume')
    need(marker['resume_checkpoint_sha256']==t['resume']['sha256'] and Path(marker['resume_checkpoint']).resolve()==resume,
         'Actual resume marker binding mismatch')
    raw=Path(marker['raw_checkpoint']).resolve();checked(raw,candidate_sha)
    state=torch.load(raw,map_location='cpu',weights_only=True,mmap=True)
    saved=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
    need(len(state)==514 and sum(v.numel() for v in state.values())==259474432 and
         all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()),'New raw is not514 finite FP32 HYPIR LoRA tensors')
    initial=torch.load(INITIAL_A,map_location='cpu',weights_only=True,mmap=True)
    need(list(initial)==list(state), 'InitialA and candidate LoRA keys differ')
    changed=sum(not torch.equal(v,initial[k]) for k,v in state.items())
    need(changed>0 and changed==marker['changed_FP32_LoRA_tensors'], 'Actual changed tensor count differs from marker')
    need(saved['protocol_id']==marker['protocol_id'] and saved['local_step']==saved['optimizer_step']==local and saved['global_step']==global_step,
         'Actual immutable resume counters differ')
    need(list(state)==list(saved['lora']) and all(torch.equal(v,saved['lora'][k]) for k,v in state.items()), 'Actual raw/resume LoRA data differ')
    need(len(saved['optimizer']['state'])==514 and {int(v['step']) for v in saved['optimizer']['state'].values()}=={local}, 'Actual Adam step is not new-stage local step')
    groups=saved['optimizer']['param_groups'];ids=[i for g in groups for i in g['params']]
    need(len(groups)==1 and len(ids)==len(set(ids))==514 and set(ids)==set(saved['optimizer']['state']), 'Adam parameter IDs differ')
    g=groups[0]
    need(g['lr']==p['objective']['lr']==1e-6 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 and g['foreach'] is False, 'Fresh Adam hyperparameters differ')
    need(all(v[k].dtype==torch.float32 and bool(torch.isfinite(v[k]).all()) for v in saved['optimizer']['state'].values() for k in ['exp_avg','exp_avg_sq']),
         'Actual Adam moments nonfinite or notFP32')
    need(all(k in saved for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']), 'Training resume must retain4 RNG states')
    parity=Path(p['deployment_parity_proof']);checked(parity,p['deployment_parity_proof_sha256']);proof=read(parity)
    need(proof['status']=='PASS_DEPLOYMENT_FUSED_PARITY_AND_514_GRADIENTS' and proof['raw_sha256']==INITIAL_A_SHA and
         proof['module_sha256']==p['deployment_aligned_module_sha256'] and
         proof['script_sha256']=='1153ebabb7047c60b6fa510ab81599800dba064b14201a85a6fb7b19920acf73' and
         proof['all_fourRNG_unchanged'] is True and proof['all_fourRNG_before']==proof['all_fourRNG_after'] and
         proof['master_weights_unchanged'] is True and proof['merged_weight_exact_count']==257 and proof['eager_fused_UNet_forward']['exact'] is True and
         proof['gradient_FP32_finite_nonzero_count']==514,'Real deployment-forward/gradient parity proof notPASS')
    return dict(initial_checkpoint_sha256=INITIAL_A_SHA,global_step=global_step,local_step=local,optimizer_step=local,
                fresh_Adam_initial_step=0,no_teacher_outputs=True,no_val_GT=True,input_count=100,
                input_catalog=ref(inputs_path),training_config=ref(config_path),marker=ref(marker_path),resume=ref(resume),
                raw_and_resume514_FP32_lora_equal=True,actual_changed_FP32_LoRA_tensors=changed,actual_Adam514_FP32_finite=True,actual_Adam_param_IDs_and_hyperparameters_checked=True,training_resume4RNG_present=True,
                deployment_parity_proof=ref(parity),training_sources=[dict(path=k,sha256=v) for k,v in p['source_sha256'].items()])


def audit_selected(evaluation,model):
    global torch,np
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(4)
    need(not torch.cuda.is_initialized(),'CPU audit initialized CUDA')
    mp=model.parent/'candidate_manifest.json';m=read(mp);c=read(model/'config.json')
    need(Path(m['model_dir']).resolve()==model and m['label']==c['label'],'Manifest runtime/label mismatch')
    need(m['model_type']=='HYPIR_SD2_rank256_native512' and m['no_teacher_outputs'] is True and
         m['no_val_GT_training'] is True,'Candidate must remain directly adapted HYPIR, no distillation/GT training')
    checked(model/'runner.py',RUNNER_SHA);need(m['runner_sha256']==RUNNER_SHA,'Must retain frozen native512 Push6 HYPIR runner')
    checked(model/'config.json',m['config_sha256'])
    parent_config=read(PARENT_MODEL/'config.json')
    need({k:v for k,v in c.items() if k not in ['label','weight_path']}==
         {k:v for k,v in parent_config.items() if k not in ['label','weight_path']},'Runtime config changed beyond label/new LoRA checkpoint path')
    checkpoint=Path(c['weight_path']).resolve();need(Path(m['checkpoint']).resolve()==checkpoint,'Manifest checkpoint path mismatch')
    need(m['checkpoint_sha256']!=INITIAL_A_SHA and m['initial_checkpoint_sha256']==INITIAL_A_SHA,
         'Candidate/initialA provenance must be separate')
    checked(checkpoint,m['checkpoint_sha256']);checked(INITIAL_A,INITIAL_A_SHA)
    checked(PARENT_AUDIT,PARENT_AUDIT_SHA)
    training=training_binding(m,m['checkpoint_sha256'])
    need(all(m[k]==training[k] for k in ['global_step','local_step','optimizer_step']), 'Manifest training counters differ from actual marker/resume')
    need(Path(read(training['marker']['path'])['raw_checkpoint']).resolve()==checkpoint,'Marker is for another raw checkpoint')
    for name in ['render_status.json','render_protocol.json','render_manifest.json','render_progress.pt',
                 'scores/status.json','scores/protocol.json','scores/result.json','scores/measurements.jsonl','scores/rng_checkpoint.pt']:sha(evaluation/name)
    need((evaluation/'render_complete').is_file(),'Formal render not complete')
    result=read(evaluation/'scores/result.json')
    need(result['official_full_count'] is True and result['completed']==result['expected']==103,'Require actual fresh full103/418, never a pilot')
    rp=read(evaluation/'render_protocol.json');p=rp['protocol'];rr=p['runner']
    need(Path(p['model_dir']).resolve()==model and rr['runner_sha256']==RUNNER_SHA and rr['checkpoint_sha256']==m['checkpoint_sha256'] and
         Path(rr['weight_path']).resolve()==checkpoint,'Formal render runtime/checkpoint mismatch')
    need(p['model_dir_sha256']['config.json']==m['config_sha256'] and rr['label']==c['label'],'Rendered config/label is not selected candidate')
    need(p['schema']=='cuda_submission_quality_v1' and p['cases']=='all' and p['warmup'] is False and
         p['source_sha256']==RENDER_SHA and p['inference_rng']=='seed231 before model initialization; continuous ordered val3 then tests',
         'Original renderer/rng/order scope changed')
    need(rr['model_t']==200 and rr['coeff_t']==175 and rr['lora_rank']==256 and rr['patch']==512 and rr['stride']==256 and
         rr['scale']==1 and rr['caption']=='' and rr['internal_precision']=='BF16' and
         rr['seed']=='231 before initialization; continuous inference RNG','Not the original native512 HYPIR protocol')
    checked(R/'code/render_cuda_submission.py',RENDER_SHA);checked(R/'code/score_cuda_stream.py',SCORER_SHA);checked(R/'scoring_tool/csig_metrics.py',CM_SHA)
    manifest=read(evaluation/'render_manifest.json');need(len(p['tasks'])==len(manifest['records'])==103,'Formal inputs/PNG records not103')
    for index,((split,case),task,row) in enumerate(zip(ORDER,p['tasks'],manifest['records'])):
        src=R/'source_payload/scoring_assets/data/验证集'/f'case{case}_lq.jpg' if split=='val' else R/'source_payload/assets/input'/f'case{case}.jpg'
        relative=f'render/{split}/case{case}.png'
        need((task['split'],task['case'])==(split,case) and Path(task['input']).resolve()==src.resolve(),'Formal input/order changed')
        need((row['task_index'],row['split'],row['case'],row['output'])==(index,split,case,relative),'Formal PNG order/path changed')
        need(not (evaluation/relative).is_symlink(),'Actual canonical PNG file required')
        with Image.open(src) as image:need(list(image.size)==task['size'],'Actual input size changed')
    need({str(x.relative_to(evaluation)) for x in (evaluation/'render').rglob('*') if x.is_file()}=={r['output'] for r in manifest['records']},'Extra/missing render files')
    core=audit_core(evaluation.name,evaluation.parent)
    need(core['image_rows']==103 and core['scalar_metrics']==418 and core['actual_checkpoint_sha256']==m['checkpoint_sha256'], 'Fresh full audit/checkpoint mismatch')
    parent=read(PARENT_AUDIT)
    for task,actual,old in zip(p['tasks'],core['records'],parent['records']):
        need((actual['split'],actual['case'])==(old['split'],old['case']) and task['input_sha256']==old['actual_input_sha256'] and
             actual['actual_gt_sha256']==old['actual_gt_sha256'],'Formal dataset/GT differs from frozenA comparison cohort')
    rows=[json.loads(s) for s in (evaluation/'scores/measurements.jsonl').read_text().splitlines() if s.strip()]
    for row in rows:
        need(row['seed_scope']=='split_seed1234_continuous_rng' and abs(nr(row['metrics'])-row['nr_per_image'])<=1e-12,'Per-image original NR/RNG metadata mismatch')
        if row['split']=='test':need(row['gt_sha256'] is None,'Test row must not contain GT')
    sp=read(evaluation/'scores/protocol.json');psp=read(PARENT/'scores/protocol.json')
    for key in ['metric_source','original_tool_deployment_sha256','original_sources','script_sha256','environment_protocol_sha256','environment_versions',
                'numpy_legacy_compat','seed','seed_scope','metric_order','image_policy','resume_policy','interleaving_policy','sequence']:
        need(sp[key]==psp[key],'Original scorer metadata differs: '+key)
    proofs=[json.loads(json.dumps(q['maniqa_offline_init_proof'])) for q in [sp,psp]]
    for q in proofs:
        for row in q['before_init_records']:row.pop('before_original_checkpoint_load',None)
    need(proofs[0]==proofs[1] and core['score_post_rng_hashes']==parent['score_post_rng_hashes'], 'Metric weights/crops/eval or continuous score RNG changed')
    parent_render=read(PARENT/'render_protocol.json')['protocol']
    for key in ['schema','cases','input_encoding','output_encoding','warmup','inference_rng','source_sha256','torch']:
        need(p[key]==parent_render[key],'Formal renderer encoding changed: '+key)
    core.update(schema='hypir_direct_nr_full_audit_v2',quality_mode='fresh_original103_418',quality_reused=False,
        model_dir=str(model),candidate_manifest=ref(mp),config=ref(model/'config.json'),runner=ref(model/'runner.py'),
        initial_checkpoint_sha256=INITIAL_A_SHA,training_provenance=training,auditor_source_sha256=sha(__file__),
        global_step=training['global_step'],local_step=training['local_step'],optimizer_step=training['optimizer_step'],
        actual_render_files=103,extra_render_files=0,no_teacher_outputs=True,no_val_GT_training=True,
        render_RNG_comparison_to_initialA_required=False,metric_RNG_matches_frozen_A=True,no_latency_claim=True,no_official_prediction=True)
    stable_files();return core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    d=a.evaluation_dir.resolve();model=a.model_dir.resolve();out=a.out.resolve()
    need(out.is_relative_to(R/'audits/push7'),'Use a new Push7 audit output')
    need(not out.exists(),'Immutable audit exists; use new label')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',evaluation_dir=str(d),model_dir=str(model),out=str(out),
            raw='actual new LoRA checkpoint SHA from candidate manifest',initialA=INITIAL_A_SHA,
            runtime_runner_sha256=RUNNER_SHA,scope='fresh full103/418, no teacher outputs/no valGT training; no candidate quality reuse',GPU=False)))
        return
    q=audit_selected(d,model);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(q,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=q['status'],out=str(out),sha256=sha(out),NR=q['NR'],FR=q['FR'],S_proxy=q['S_proxy'])))


if __name__=='__main__':main()
