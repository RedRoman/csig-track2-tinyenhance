"""Independently join final full103 Push3 quality with three fresh paired timings.

CPU-only standard library, apart from importing SHA-pinned original predict.py
(which itself uses only argparse). Never starts benchmarks or imports torch.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import math

ROOT = Path('/cache/independent_sr_cuda_20261013')
PREDICT_SHA = '0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
BENCHMARK_SHA = '3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995'
CM_SHA = '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
PARENT_SHA = 'cfcc3013fa87aa721f346d6cc285e70b6d6ff2c88315ce9f1465b07ffe020e68'
CACHE = {}; TOKENS = {}


def need(ok, message):
    if not ok:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text())
def token(path):
    s = Path(path).stat();return s.st_size, s.st_mtime_ns, s.st_ino


def sha(path):
    p = Path(path).resolve(); before = token(p)
    if p not in CACHE or CACHE[p][0] != before:
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):h.update(block)
        need(token(p) == before, f'Source changed while hashing: {p}')
        CACHE[p] = before, h.hexdigest()
    TOKENS[str(p)] = before
    return CACHE[p][1]


def check_file(path, digest):
    need(sha(path) == digest, f'Actual source/artifact SHA differs: {path}')


def identity(value):return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def full_quality(selection, audit_path):
    directory = Path(selection['quality_dir']).resolve()
    need(directory.is_relative_to(ROOT), 'Quality source outside experiment root')
    for relative in ['render_manifest.json', 'render_protocol.json', 'scores/measurements.jsonl',
                     'scores/protocol.json', 'scores/result.json']:
        sha(directory / relative)
    audit = read(audit_path)
    check_file(audit_path, selection['full_audit_sha256'])
    need(audit['status'] == 'PASS' and audit['image_rows'] == 103 and audit['scalar_metrics'] == 418,
         'Independent full quality audit must PASS103/418')
    need(audit['label'] == selection['label'] and audit['actual_checkpoint_sha256'] == selection['checkpoint_sha256'],
         'Independent audit label/checkpoint mismatch')
    need(audit['actual_png_manifest_scored_sha'] == 'PASS103' and audit['actual_source_code_sha'] == 'PASS', 'Actual source/PNG audit absent')
    need((directory / 'render_complete').is_file(), 'Full renderer incomplete')
    result = read(directory / 'scores/result.json')
    need(result['status'] == 'complete' and result['completed'] == result['expected'] == 103 and result['official_full_count'] is True,
         'Full quality result incomplete')
    need(result['S_proxy_scope'] == 'full100_test_NR_plus3val_FR' and result['validation']['n'] == 3 and result['test']['n'] == 100,
         'Full quality split/scope differs')
    manifest = read(directory / 'render_manifest.json'); rp = read(directory / 'render_protocol.json')
    need(manifest['protocol'] == rp['protocol'] and manifest['protocol_id'] == rp['protocol_id'] == identity(rp['protocol']),
         'Render protocol identity mismatch')
    p = rp['protocol']; runner = p['runner']; model = Path(selection['model_dir']).resolve()
    need(Path(p['model_dir']).resolve() == model and runner['checkpoint_sha256'] == selection['checkpoint_sha256']
         and Path(runner['weight_path']).resolve() == Path(selection['checkpoint']).resolve(), 'Render runner/checkpoint mismatch')
    check_file(selection['checkpoint'], selection['checkpoint_sha256'])
    check_file(model / 'runner.py', runner['runner_sha256'])
    for relative, digest in runner['source_sha256'].items():
        check_file(Path(runner['repo_path']) / relative, digest)
    for relative, digest in p['model_dir_sha256'].items():
        if '__pycache__' not in Path(relative).parts and not relative.endswith('.pyc'):check_file(model / relative, digest)
    renderer = Path(selection.get('renderer_source', ROOT / 'code/render_cuda_submission.py'))
    check_file(renderer, p['source_sha256'])
    expected = [('val', i) for i in range(1, 4)] + [('test', i) for i in range(1, 101)]
    records = manifest['records']
    need([(r['split'], r['case']) for r in records] == expected and len({r['output'] for r in records}) == 103,
         'Full render does not contain unique ordered val3/test100')
    need([(r['split'], r['case']) for r in p['tasks']] == expected, 'Render input sequence differs')
    rows = [json.loads(line) for line in (directory / 'scores/measurements.jsonl').read_text().splitlines() if line.strip()]
    need([(r['split'], r['case']) for r in rows] == [('validation' if split == 'val' else split, n) for split, n in expected],
         'Full metric row sequence differs')
    for row, record, task in zip(rows, records, p['tasks']):
        check_file(directory / record['output'], record['sha256'])
        check_file(task['input'], task['input_sha256'])
        need(row['sr_sha256'] == record['sha256'] and Path(row['sr_path']).resolve() == (directory / record['output']).resolve(),
             'Scored image differs from actual rendered PNG')
    score_protocol = read(directory / 'scores/protocol.json')
    check_file(ROOT / 'code/score_cuda_stream.py', score_protocol['script_sha256'])
    for source in score_protocol['original_sources']:check_file(source['resolved_path'], source['sha256'])
    need(next(r['sha256'] for r in score_protocol['original_sources'] if r['path'] == 'csig_metrics.py') == CM_SHA,
         'Unmodified original csig_metrics source required')
    for relative, record in audit['actual_metric_weights_sha256'].items():
        check_file(ROOT / 'scoring_tool/weights' / relative, record['sha256'])
    means = {}; scalars = 0
    for split, n, keys in [('validation', 3, ['lpips', 'dists', 'niqe', 'musiq', 'maniqa', 'clipiqa']),
                           ('test', 100, ['niqe', 'musiq', 'maniqa', 'clipiqa'])]:
        group = [r for r in rows if r['split'] == split];need(len(group) == n, 'Metric split count differs')
        need(all(set(r['metrics']) == set(keys) for r in group), 'Unexpected metric components')
        means[split] = {}
        for key in keys:
            values = [r['metrics'][key] for r in group]
            need(all(math.isfinite(v) for v in values), 'Nonfinite actual metric scalar')
            value = sum(values) / n; means[split][key] = value
            need(abs(result[split][key] - value) < 1e-12, 'Published metric mean differs')
        scalars += n * len(keys)
    need(scalars == 418, 'Wrong full scalar count')
    t, v = means['test'], means['validation']
    nr = t['clipiqa'] + t['maniqa'] + t['musiq'] / 100 + max(0, (10 - t['niqe']) / 10)
    fr = 2 - v['lpips'] - v['dists']; total = nr + fr
    for key, value in [('NR', nr), ('FR', fr), ('S_proxy', total)]:
        need(abs(selection[key] - value) < 1e-12 and abs(audit[key] - value) < 1e-12, f'Selection/audit {key} differs')
    need(abs(result['test']['nr'] - nr) < 1e-12 and abs(result['validation']['fr'] - fr) < 1e-12
         and abs(result['S_proxy'] - total) < 1e-12, 'Original NR/FR/S formula differs')
    return {'label': selection['label'], 'NR': nr, 'FR': fr, 'S': total, 'component_means': means,
            'checkpoint_sha256': selection['checkpoint_sha256'], 'model_dir': str(model),
            'evaluation_dir': str(directory), 'render_protocol': p,
            'result_sha256': sha(directory / 'scores/result.json'), 'independent_audit_sha256': sha(audit_path),
            'actual103_PNG_input_manifest_scored_SHA_verified': True, 'image_rows': 103, 'scalar_metrics': 418}


def percentile(values, percent):
    ordered = sorted(values); place = (len(ordered) - 1) * percent / 100
    lo = int(math.floor(place)); hi = int(math.ceil(place));f = place - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * f


def timing(path, quality=None, reference=None):
    sha(path)
    value = read(path)
    need(value['status'] == 'complete' and value['cuda']['logical_device'] == 0, 'Fresh benchmark incomplete/wrong logical GPU')
    need(value['input']['shape'] == [1, 3, 512, 512] and value['input']['dtype'] == 'torch.float16'
         and value['input']['device'] == 'cuda:0' and value['seed'] == 231, 'Benchmark input/seed differs')
    need(value['output']['all_finite'] is True and value['output']['within_minus1_plus1'] is True
         and value['output']['device'] == 'cuda:0' and value['output']['shape'] == [1, 3, 512, 512]
         and value['output']['clamping_applied_by_benchmark'] is False, 'Invalid benchmark output/range/device')
    stats = value['latency_512'];need(stats['n'] == 200 and stats['warm'] == 30, 'Benchmark must be fresh warm30/n200')
    for kind in ['event', 'wall']:
        samples = value['raw_samples'][kind + '_ms']
        need(len(samples) == 200 and all(math.isfinite(x) and x > 0 for x in samples), 'Invalid original timing sample vectors')
        for key, calculated in [('p10', percentile(samples, 10)), ('med', percentile(samples, 50)), ('min', min(samples))]:
            need(abs(stats[kind + '_' + key] - calculated) < 1e-12, 'Raw-sample timing statistic differs')
    need(value['benchmark_script']['sha256'] == BENCHMARK_SHA, 'Original benchmark SHA differs')
    files = value['artifacts']['files'] + value['artifacts']['extra_files'] + [value['benchmark_script']]
    for record in files:
        p = Path(record['resolved_path']);need(p.stat().st_size == record['bytes'], 'Actual timing artifact size differs')
        check_file(p, record['sha256'])
    if quality is not None:
        p = quality['render_protocol'];runner = p['runner']
        need(value['model_dir'] == p['model_dir'] and Path(value['model_dir']).resolve() == Path(quality['model_dir']),
             'Latency measures a different model directory')
        need(any(r['sha256'] == runner['checkpoint_sha256'] and Path(r['resolved_path']).resolve() == Path(runner['weight_path']).resolve()
                 for r in value['artifacts']['extra_files']), 'Latency must explicitly hash exact full-quality checkpoint as extra-file')
        need(any(r['path'] == 'runner.py' and r['sha256'] == runner['runner_sha256'] for r in files), 'Latency runner differs from full-quality runner')
        need(any(r['path'] == 'config.json' and r['sha256'] == p['model_dir_sha256']['config.json'] for r in files), 'Latency config differs from full-quality config')
    else:
        need(Path(value['model_dir']).resolve() == ROOT / 'calibration/g4/model_dir', 'Fresh reference must use original G4 model')
    if reference is not None:
        ref = reference['latency_512']['event_p10']
        need(value['cuda'] == reference['cuda'] and value['ref_ms'] == ref and value['anchor_g4_ms'] == .431,
             'G4 reference/hardware/runtime differs')
        need(abs(value['g_used_ms'] - stats['event_p10'] * .431 / ref) < 1e-12, 'Same-card normalization differs')
    return value


def launch_proof(path, calibrations, selection):
    sha(path)
    launch = read(path)
    need(launch['status'] == 'complete' and launch['schema'] == 'push3_fresh_paired_latency_launch_v1'
         and launch['same_UUID_PCI_all_three'] is True and launch['cuda_visible_devices'] == '0', 'Fresh paired launch provenance incomplete')
    need([row['key'] for row in launch['runs']] == ['g4', 'parent6500', 'selected'], 'Fresh subprocess run order differs')
    physical = launch['initial_physical_gpu']
    need(physical['index'] == 0 and physical['uuid'].startswith('GPU-') and not physical['active_compute_processes'], 'Initial physical GPU0 was not recorded idle')
    need(launch['candidate']['checkpoint_sha256'] == selection['checkpoint_sha256']
         and launch['candidate']['model_dir'] == selection['model_dir'], 'Fresh launch candidate differs from final selection')
    need(launch['benchmark_sha256'] == BENCHMARK_SHA, 'Launch benchmark source mismatch')
    check_file(ROOT / 'code/push3/paired_latency_push3.py', launch['launcher_sha256'])
    for row in launch['runs']:
        need(row['returncode'] == 0 and row['cuda_visible_devices'] == '0', 'Paired subprocess failed/wrong CUDA selection')
        out = ROOT / f'calibration/push3_{row["key"]}_latency.json'
        need(Path(row['out']) == out and sha(out) == row['out_sha256'], 'Fresh output differs from recorded launch')
        for moment in ['physical_before', 'physical_after']:
            p = row[moment]
            need((p['index'], p['uuid'], p['pci_bus_id']) == (0, physical['uuid'], physical['pci_bus_id']), 'Different physical UUID/PCI bus across timings')
        need(not row['physical_before']['active_compute_processes'], 'GPU0 occupied before a paired benchmark')
        need(row['argv'][1] == str(ROOT / 'code/benchmark_cuda.py'), 'Subprocess benchmark path differs')
        for flag, expected in [('--device', '0'), ('--warm', '30'), ('--n', '200'), ('--seed', '231')]:
            need(row['argv'][row['argv'].index(flag) + 1] == expected, 'Launch options differ from paired protocol')
    return {'uuid': physical['uuid'], 'pci_bus_id': physical['pci_bus_id'], 'name': physical['name'],
            'same_physical_UUID_PCI_all_three': True, 'fresh_sequential_processes': True,
            'launch_json_sha256': sha(path)}


def join(selection_path):
    selection_sha = sha(selection_path)
    selection = read(selection_path)
    need(selection['status'] == 'selected_after_independent_full_audit', 'Final independently audited selection required')
    sha(ROOT / 'results/nr_push2_selection.json')
    parent = read(ROOT / 'results/nr_push2_selection.json')
    need(parent['global_step'] == 6500 and parent['checkpoint_sha256'] == PARENT_SHA, 'Sealed best6500 parent required')
    parent_audit = ROOT / 'audits/push2/full/nr_stage3_anchor6250_m15_s6500_compiled_all_audit.json'
    parent_quality = full_quality(parent, parent_audit)
    selected_quality = full_quality(selection, Path(selection['full_audit']))
    source = ROOT / 'scoring_tool/predict.py';check_file(source, PREDICT_SHA)
    spec = importlib.util.spec_from_file_location('_unchanged_original_csig_predict', source)
    original = importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    g4 = timing(ROOT / 'calibration/push3_g4_latency.json')
    parent_timing = timing(ROOT / 'calibration/push3_parent6500_latency.json', parent_quality, g4)
    selected_timing = timing(ROOT / 'calibration/push3_selected_latency.json', selected_quality, g4)
    physical = launch_proof(ROOT / 'calibration/push3_latency_launch.json', [g4, parent_timing, selected_timing], selection)
    parent_g, selected_g = parent_timing['g_used_ms'], selected_timing['g_used_ms']
    parent_prediction = original.predict(parent_quality['S'], parent_g)
    selected_prediction = original.predict(selected_quality['S'], selected_g)
    fixed_latency_prediction = original.predict(selected_quality['S'], parent_g)
    quality_factor = 2 ** (selected_quality['S'] - parent_quality['S'])
    timing_factor = ((original.A + original.K * parent_g) / (original.A + original.K * selected_g)) ** .2
    need(abs(selected_prediction / parent_prediction - quality_factor * timing_factor) < 1e-12, 'Original prediction factor identity differs')
    for path, before in TOKENS.items():need(token(path) == before, f'Actual source changed before join completion: {path}')
    return {'schema': 'push3_fresh_paired_full103_prediction_v1', 'status': 'PASS',
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'CPU_only': True,
            'selection_sha256': selection_sha, 'original_predict_sha256': PREDICT_SHA,
            'benchmark_script_sha256': BENCHMARK_SHA, 'G4_event_p10_ms': g4['latency_512']['event_p10'],
            'physical_card_proof': physical, 'parent6500_quality': parent_quality,
            'selected_quality': selected_quality, 'parent6500_event_p10_ms': parent_timing['latency_512']['event_p10'],
            'selected_event_p10_ms': selected_timing['latency_512']['event_p10'],
            'parent6500_normalized_g_ms': parent_g, 'selected_normalized_g_ms': selected_g,
            'parent6500_current_paired_prediction': parent_prediction, 'selected_current_paired_prediction': selected_prediction,
            'selected_parent_latency_fixed_quality_only_prediction': fixed_latency_prediction,
            'delta_NR_vs_parent6500': selected_quality['NR'] - parent_quality['NR'],
            'delta_FR_vs_parent6500': selected_quality['FR'] - parent_quality['FR'],
            'delta_S_vs_parent6500': selected_quality['S'] - parent_quality['S'],
            'quality_factor_vs_parent6500': quality_factor,
            'measured_timing_factor_not_proven_algorithm_speedup': timing_factor,
            'combined_factor_vs_parent6500': quality_factor * timing_factor,
            'calibration_sha256': {key: sha(ROOT / f'calibration/push3_{key}_latency.json') for key in ['g4', 'parent6500', 'selected']},
            'interpretation': 'Original five-point empirical tool prediction, not official measured leaderboard score. Both parent and selected use this same fresh G4 calibration. Fixed-parent-latency quality-only change is reported separately. One run and tiny timing differences do not establish a stable algorithm speedup.',
            'no_historical_prediction_reused': True, 'timing_and_quality_sources_checked_unchanged_before_join': True}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection-json', type=Path, default=ROOT / 'results/nr_push3_selection.json')
    ap.add_argument('--out', type=Path, default=ROOT / 'calibration/push3_paired_prediction.json')
    args = ap.parse_args(); report = join(args.selection_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
