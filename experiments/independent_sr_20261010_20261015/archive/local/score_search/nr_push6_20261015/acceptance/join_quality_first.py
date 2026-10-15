"""CPU-only Push6 fresh-triplet/quality join. Default is a plan; no GPU imports.

Until an independent exact103 or rescored103 proof passes, publish timing only
and PENDING_QUALITY, with no candidate NR/FR/prediction fields.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import struct
import sys
import tempfile

sys.dont_write_bytecode = True
R = Path('/cache/independent_sr_cuda_20261013')
D = R / 'calibration/push6_quality_first'
PARENT = R / 'nr_push4_full_evaluations/push4_a7750_compiled_all'
CAND = R / 'nr_push6_full_evaluations/push6_a7750_ro_wavelet_all'
RAW = R / 'adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth'
RAW_SHA = '589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
BENCH_SHA = '3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
PRED_SHA = '0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
COMPARE_SHA = 'd93e3aa9792e3cdab6b3933594f06d17e0a9d7ab12e62fee44570159cefecef8'
RENDER_SHA = '1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
SCORER_SHA = '20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074'
CM_SHA = '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
ORDER = [('val', i) for i in range(1, 4)] + [('test', i) for i in range(1, 101)]
MODELS = {
 'g4': ('g4_latency.json', R/'calibration/g4/model_dir', '6222ead92dc6fb24b88c0350e41191fc27a5adc2f6ba38bd383a4badb9bd8c1e'),
 'parent': ('a7750_parent_latency.json', R/'submissions/hypir_push4_a7750_compiled/model_dir', '396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae'),
 'candidate': ('a7750_ro_wavelet_latency.json', R/'submissions/hypir_push6_a7750_ro_wavelet/model_dir', 'dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b')}
SEEN = {}


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
         and x['input']['device'] == 'cuda:0' and x['cuda']['logical_device'] == 0, 'Timing CUDA input differs')
    y = x['output']
    need(y['device'] == 'cuda:0' and y['shape'] == [1,3,512,512] and y['all_finite'] is True
         and y['within_minus1_plus1'] is True and -1 <= y['min'] <= y['max'] <= 1
         and y['clamping_applied_by_benchmark'] is False, 'Timing output invalid')
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
    need(x['schema'] == 'push6_root_launch_evidence_v1' and x['serial'] is True
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


def cohort(directory, key):
    from PIL import Image
    need((directory/'render_complete').is_file(), 'Formal full not complete')
    s = read(directory/'render_status.json'); m = read(directory/'render_manifest.json'); rp = read(directory/'render_protocol.json')
    p = m['protocol']; ident = hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    need(s['status'] == 'complete' and s['completed'] == s['total'] == 103, 'Not full103')
    need(ident == m['protocol_id'] == rp['protocol_id'] and rp['protocol'] == p, 'Render identity differs')
    need(p['cases'] == 'all' and p['warmup'] is False and p['source_sha256'] == RENDER_SHA, 'Wrong formal renderer')
    checked(R/'code/render_cuda_submission.py', RENDER_SHA)
    need(Path(p['model_dir']).resolve() == MODELS[key][1] and p['runner']['runner_sha256'] == MODELS[key][2]
         and p['runner']['checkpoint_sha256'] == RAW_SHA, 'Formal model identity differs from timing')
    for name, digest in p['runner']['source_sha256'].items(): checked(Path(p['runner']['repo_path'])/name, digest)
    for name in ['runner.py','config.json']: checked(Path(p['model_dir'])/name,p['model_dir_sha256'][name])
    need(len(m['records']) == len(p['tasks']) == 103, 'Incomplete manifest/tasks')
    for (split,case), task, row in zip(ORDER,p['tasks'],m['records']):
        relative = f'render/{split}/case{case}.png'; png = directory/relative
        need((row['split'],row['case'],row['output']) == (split,case,relative)
             and (task['split'],task['case']) == (split,case), 'Manifest ordering/path differs')
        checked(png,row['sha256']); checked(task['input'],task['input_sha256'])
        with Image.open(png) as im: need(im.format == 'PNG' and im.mode == 'RGB' and list(im.size) == row['size'] == task['size'], 'Native PNG invalid')
    need({str(f.relative_to(directory)) for f in (directory/'render').rglob('*') if f.is_file()} == {r['output'] for r in m['records']}, 'Extra/missing render file')
    return m


def scored(directory, manifest):
    protocol = read(directory/'scores/protocol.json'); result = read(directory/'scores/result.json')
    rows_path = directory/'scores/measurements.jsonl'; sha(rows_path)
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    need(protocol['seed'] == 1234 and protocol['script_sha256'] == SCORER_SHA, 'Score source/seed differs')
    checked(R/'code/score_cuda_stream.py',SCORER_SHA); checked(R/'scoring_tool/csig_metrics.py',CM_SHA)
    proof = protocol['maniqa_offline_init_proof']
    need(proof['test_sample'] == 100 and proof['net_training'] is False and proof['all_final_parameters_and_buffers_equal_checkpoint'] is True, 'Not original MANIQA100/eval')
    need(len(rows) == 103 and result['status'] == 'complete' and result['completed'] == result['expected'] == 103
         and result['official_full_count'] is True and result['S_proxy_scope'] == 'full100_test_NR_plus3val_FR', 'Score not full103/418')
    frozen = None
    if directory == PARENT:
        fp = R/'audits/push4/evaluations/push4_a7750_compiled_all/audit.json'
        checked(fp,'ae5b2234f5e110240507fbec515139d1b2a23bf8f664e7477e34c377e09c507b')
        frozen = read(fp)
        need(frozen['status']=='PASS' and frozen['image_rows']==103 and frozen['scalar_metrics']==418
             and result==frozen['published_result'], 'Parent published result changed from frozen audit')
        for record, task, old in zip(manifest['records'],manifest['protocol']['tasks'],frozen['records']):
            need(record['sha256']==old['actual_png_sha256'] and task['input_sha256']==old['actual_input_sha256'], 'Parent PNG/input changed from frozen audit')
    scalars = 0
    for (split,case), row, record in zip(ORDER,rows,manifest['records']):
        need((row['split'],row['case']) == ('validation' if split=='val' else split,case)
             and row['sr_sha256'] == record['sha256'] and Path(row['sr_path']).resolve() == directory/record['output'], 'Scored PNG/order differs')
        keys = {'niqe','musiq','maniqa','clipiqa'} | ({'lpips','dists'} if split=='val' else set())
        need(set(row['metrics']) == keys and all(math.isfinite(v) for v in row['metrics'].values()), 'Metric set/value differs')
        scalars += len(keys)
        if split == 'val':
            checked(R/'source_payload/scoring_assets/data/验证集'/(f'case{case}_gt.jpg' if case==3 else f'case{case}_gt.png'),row['gt_sha256'])
            if frozen: need(row['gt_sha256']==frozen['records'][case-1]['actual_gt_sha256'], 'Parent GT changed from frozen audit')
    means = {split:{k:statistics.fmean(r['metrics'][k] for r in rows if r['split']==split) for k in rows[0 if split=='validation' else 3]['metrics']} for split in ['validation','test']}
    for split, values in means.items():
        for k,v in values.items(): need(abs(v-result[split][k]) < 1e-12, 'Mean/result differs')
    t,v = means['test'],means['validation']; nr=t['clipiqa']+t['maniqa']+t['musiq']/100+max(0,(10-t['niqe'])/10); fr=2-v['lpips']-v['dists']
    need(scalars==418 and abs(nr-result['test']['nr'])<1e-12 and abs(fr-result['validation']['fr'])<1e-12 and abs(nr+fr-result['S_proxy'])<1e-12, 'Quality formula differs')
    return dict(NR=nr,FR=fr,S_proxy=nr+fr,image_rows=103,scalar_metrics=418,score_result=ref(directory/'scores/result.json'),score_records=ref(rows_path),score_protocol=ref(directory/'scores/protocol.json'))


def quality(path, mode):
    proof = read(path)
    if mode == 'exact103':
        if proof.get('status') != 'PASS_EXACT103': return None
        need(proof['schema']=='push6_formal_png103_comparison_v2' and proof['helper_sha256']==COMPARE_SHA, 'Unknown exact103 v2 audit')
        checked(R/'code/push6/compare_formal_png103_v2.py',COMPARE_SHA)
        need(proof['PNG_exact_count']==103 and proof['final_fourRNG_equal'] is True and proof['metric_input_reuse_eligible'] is True
             and proof['actual_formal_inputs_checked']==103 and proof['actual_validation_GT_checked']==3
             and proof['actual_PNG_checked_each_cohort']==103 and proof['extra_render_files']==0, 'Exact103 gates incomplete')
        need(proof['parent_final_fourRNG']==proof['candidate_final_fourRNG'], 'Final RNG proof differs')
    else:
        if proof.get('status') != 'PASS': return None
        need(proof['image_rows']==103 and proof['scalar_metrics']==418
             and Path(proof['evaluation_dir']).resolve()==CAND and proof['actual_checkpoint_sha256']==RAW_SHA, 'Independent rescored audit binding differs')
    need(Path(proof.get('candidate_dir',proof.get('evaluation_dir'))).resolve()==CAND, 'Quality proof belongs to another cohort')
    parent = cohort(PARENT,'parent'); candidate = cohort(CAND,'candidate'); pq = scored(PARENT,parent)
    need(parent['protocol']['tasks']==candidate['protocol']['tasks'], 'Candidate input103 differs')
    if mode == 'exact103':
        checked(PARENT/'render_manifest.json',proof['parent_render_manifest_sha256']); checked(CAND/'render_manifest.json',proof['candidate_render_manifest_sha256'])
        need(all(a['sha256']==b['sha256'] for a,b in zip(parent['records'],candidate['records'])), 'Actual full103 PNG differs')
        need(proof['checkpoint_sha256']==RAW_SHA and proof['parent_runner_sha256']==MODELS['parent'][2]
             and proof['candidate_runner_sha256']==MODELS['candidate'][2], 'Exact proof runner/weight mismatch')
        cq=dict(pq)
        for key in ['NR','FR','S_proxy']: need(abs(cq[key]-proof['candidate_quality_from_exact_input_equivalence'][key])<1e-12, 'Reused quality differs')
    else:
        cq=scored(CAND,candidate)
        for key in ['NR','FR','S_proxy']: need(abs(cq[key]-proof[key])<1e-12, 'Independent score audit differs')
    return dict(mode=mode,independent_proof=ref(path),parent=pq,candidate=cq,
                parent_manifest=ref(PARENT/'render_manifest.json'),candidate_manifest=ref(CAND/'render_manifest.json'),
                actual_PNG103_each_cohort_rehashed=True,quality_reused_from_A=(mode=='exact103'))


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n'); temporary=f.name
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--timing-dir',type=Path,default=D)
    p.add_argument('--quality-proof',type=Path,default=R/'audits/push6/formal_png103/ro_wavelet_exact103_v2.json')
    p.add_argument('--quality-mode',choices=['exact103','rescored103'],default='exact103')
    p.add_argument('--out',type=Path,default=D/'paired_prediction.json'); p.add_argument('--execute',action='store_true'); a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY',timing_dir=str(a.timing_dir),quality_proof=str(a.quality_proof),quality_mode=a.quality_mode,out=str(a.out),GPU=False))); return
    need(a.timing_dir.resolve()==D and a.out.resolve().is_relative_to(R/'calibration'), 'Push6 scoped path required')
    timings={k:timing(k,a.timing_dir) for k in MODELS}; physical=launch(a.timing_dir)
    need(all(x['cuda']==timings['g4']['cuda'] for x in timings.values()), 'CUDA environment differs')
    result=dict(schema='push6_quality_first_fresh_triplet_v1',status='PENDING_QUALITY',CPU_only=True,
                created_utc=datetime.now(timezone.utc).isoformat(),timings=timings,launch=physical,
                source=ref(__file__),quality_proof_requested=str(a.quality_proof),quality_mode=a.quality_mode,
                speed_only=dict(parent_event_p10_ms=timings['parent']['recomputed']['event_p10'],candidate_event_p10_ms=timings['candidate']['recomputed']['event_p10'],
                                latency_reduction_fraction=1-timings['candidate']['recomputed']['event_p10']/timings['parent']['recomputed']['event_p10']),
                candidate_quality_confirmed=False,candidate_prediction_available=False)
    q=quality(a.quality_proof,a.quality_mode) if a.quality_proof.is_file() else None
    if q:
        predpath=R/'scoring_tool/predict.py'; checked(predpath,PRED_SHA)
        spec=importlib.util.spec_from_file_location('push6_original_predict',predpath); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        r=timings['g4']['recomputed']['event_p10']; predictions={}
        for key in ['parent','candidate']:
            s=q[key]['S_proxy']; t=timings[key]['recomputed']['event_p10']; g=t*.431/r
            predictions[key]=dict(NR=q[key]['NR'],FR=q[key]['FR'],S_proxy=s,event_p10_ms=t,normalized_g_ms=g,original_tool_prediction=module.predict(s,g))
        pa,ca=predictions['parent'],predictions['candidate']; quality_factor=2**(ca['S_proxy']-pa['S_proxy'])
        speed_factor=((.49+2.17*pa['normalized_g_ms'])/(.49+2.17*ca['normalized_g_ms']))**.2
        result.update(status='PASS',quality=q,predict_source=ref(predpath),predictions=predictions,
                      prediction_ratio=ca['original_tool_prediction']/pa['original_tool_prediction'],quality_factor=quality_factor,speed_factor=speed_factor,
                      candidate_quality_confirmed=True,candidate_prediction_available=True,target70=70,reaches_target=ca['original_tool_prediction']>=70,
                      score_scope='Original supplied predictor, not an actual official leaderboard submission')
    for path,(stamp,digest) in SEEN.items(): need(token(path)==stamp,'Source changed before publish: '+str(path))
    if a.out.exists() and json.loads(a.out.read_text()).get('status')=='PASS': need(result['status']=='PASS','Refuse replacing final PASS with pending')
    atomic(a.out,result)
    print(json.dumps(dict(status=result['status'],out=str(a.out),sha256=sha(a.out),speed_only=result['speed_only'],candidate_prediction_available=result['candidate_prediction_available'])))


if __name__=='__main__': main()
