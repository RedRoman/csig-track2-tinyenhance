"""Read-only CPU checks; copies only audit evidence into the authorized archive."""
from pathlib import Path
import base64
import datetime
import hashlib
import json
import os
import numpy as np

R = Path('/cache/independent_sr_cuda_20261013')
A = R / 'audits/push2/latency'
CACHE = {}; COPIES = {}


def sha(path):
    path = Path(path).resolve()
    if path not in CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(8388608), b''):
                h.update(b)
        CACHE[path] = h.hexdigest()
    return CACHE[path]


def copy(source, relative):
    source = Path(source); target = A / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes(); target.write_bytes(data)
    assert sha(source) == hashlib.sha256(target.read_bytes()).hexdigest()
    COPIES[relative] = dict(base64=base64.b64encode(data).decode(), sha256=sha(source), source=str(source))


def main():
    os.nice(10); A.mkdir(parents=True, exist_ok=True)
    keys = ['g4', 'parent6250', 'stage3_m1', 'stage3_m15']
    calibrations = {k: json.loads((R / f'calibration/push2_{k}_latency.json').read_text()) for k in keys}
    ref = calibrations['g4']['latency_512']['event_p10']; checks = {}
    for key, c in calibrations.items():
        assert c['status'] == 'complete' and c['cuda'] == calibrations['g4']['cuda']
        assert c['cuda']['logical_device'] == 0
        assert c['input']['shape'] == [1,3,512,512] and c['input']['dtype'] == 'torch.float16'
        assert c['output']['all_finite'] and c['output']['within_minus1_plus1']
        assert not c['output']['clamping_applied_by_benchmark']
        stats = c['latency_512']; raw = c['raw_samples']['event_ms']
        assert stats['n'] == len(raw) == 200 and stats['warm'] == 30
        assert abs(float(np.percentile(raw,10)) - stats['event_p10']) < 1e-12
        assert abs(float(np.median(raw)) - stats['event_med']) < 1e-12
        assert min(raw) == stats['event_min']
        records = c['artifacts']['files'] + c['artifacts']['extra_files'] + [c['benchmark_script']]
        for record in records:
            p = Path(record['resolved_path']); assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256']
            if p.suffix in ['.py','.json','.yaml','.yml','.txt'] and p.stat().st_size < 1000000:
                copy(p, 'small_artifacts/' + key + '/' + p.name)
        if key != 'g4':
            assert c['ref_ms'] == ref and c['anchor_g4_ms'] == .431
            assert abs(c['g_used_ms'] - stats['event_p10'] * .431 / ref) < 1e-12
        copy(R / f'calibration/push2_{key}_latency.json', f'push2_{key}_latency.json')
        checks[key] = dict(actual_artifact_SHA_pass=True, event_p10_ms=stats['event_p10'],
            event_median_ms=stats['event_med'], event_p90_ms=float(np.percentile(raw,90)),
            n=stats['n'], warm=stats['warm'], normalized_g_ms=c.get('g_used_ms'),
            benchmark_script_sha256=c['benchmark_script']['sha256'], cuda=c['cuda'])
    predict = R / 'scoring_tool/predict.py'
    assert sha(predict) == '0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
    copy(predict, 'sources/predict.py')
    for name in ['base_model_provenance_shared.json']:
        p = R / 'calibration' / name
        if p.exists(): copy(p, 'referenced_manifests/' + name)
    paths = [R/'nr_full_evaluations/nr_conservative_s6250_compiled_all']
    paths += [R/'nr_push2_full_evaluations'/name for name in ['nr_stage2_cont_s6500_compiled_all',
              'nr_stage3_anchor6250_m1_s6500_compiled_all','nr_stage3_anchor6250_m15_s6500_compiled_all']]
    quality = {}
    for E in paths:
        result = json.loads((E/'scores/result.json').read_text())
        if result['status'] != 'complete':
            quality[E.name] = dict(status='not_complete', prediction_eligible=False)
            continue
        assert result['completed'] == result['expected'] == 103 and result['official_full_count']
        assert result['test']['n'] == 100 and result['validation']['n'] == 3
        rows = [json.loads(x) for x in (E/'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
        m = json.loads((E/'render_manifest.json').read_text()); p = m['protocol']; runner = p['runner']
        assert len(rows) == len(m['records']) == 103 and sum(len(x['metrics']) for x in rows) == 418
        for row, rec, inp in zip(rows, m['records'], p['tasks']):
            assert row['sr_sha256'] == rec['sha256'] == sha(E/rec['output'])
            assert sha(inp['input']) == inp['input_sha256']
        assert sha(runner['weight_path']) == runner['checkpoint_sha256']
        matching = [c for c in calibrations.values() if c['model_dir'] == p['model_dir']]
        if matching:
            c = matching[0]; files = c['artifacts']['files'] + c['artifacts']['extra_files']
            assert any(x['sha256'] == runner['checkpoint_sha256'] for x in files)
            assert any(x['path']=='runner.py' and x['sha256']==runner['runner_sha256'] for x in files)
            assert any(x['path']=='config.json' and x['sha256']==p['model_dir_sha256']['config.json'] for x in files)
        for rel in ['render_manifest.json','render_protocol.json','render_status.json','scores/status.json',
                    'scores/result.json','scores/protocol.json','scores/measurements.jsonl']:
            copy(E/rel, 'quality_records/' + E.name + '/' + rel)
        quality[E.name] = dict(status='complete103', actual103_png_and_input_SHA_pass=True,
            checkpoint_sha256=runner['checkpoint_sha256'], fresh_latency_identity_matched=bool(matching))
    report = dict(status='passed', audited_bjt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        cpu_only=True, no_GPU_or_benchmark_or_original_source_mutation=True, calibrations=checks,
        quality_artifacts=quality, reference_event_p10_ms=ref, original_predict_sha256=sha(predict),
        same_card_scope='All four JSONs record identical A10080GB/logical0/runtime. Same physical GPU0 is supplied by launch provenance; UUID was not recorded.',
        timing_noise_policy='P10/median/P90 are descriptive samples, not a repeated-run confidence interval or proof of model speedup.',
        binary_weights='Actual bytes/SHA checked; binaries not duplicated here. Existing model/checkpoint backup references retained.')
    target = A/'calibration_validation.json'; target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    copy(target, 'calibration_validation.json')
    print(json.dumps(dict(audit=report, files=COPIES),ensure_ascii=False))


if __name__ == '__main__': main()
