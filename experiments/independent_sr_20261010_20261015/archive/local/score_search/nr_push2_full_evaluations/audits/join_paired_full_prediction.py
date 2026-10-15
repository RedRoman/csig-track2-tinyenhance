"""CPU-only original-tool prediction; incomplete/pilot quality never joins."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

PREDICT_SHA = '0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1'
CM_SHA = '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
LABELS = {'parent6250': 'nr_conservative_s6250_compiled_all',
          'stage3_m1': 'nr_stage3_anchor6250_m1_s6500_compiled_all',
          'stage3_m15': 'nr_stage3_anchor6250_m15_s6500_compiled_all'}


def js(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate(root, label):
    for base in ['nr_push2_full_evaluations/results', 'nr_push2_full_evaluations',
                 'nr_cuda_evaluations', 'nr_full_evaluations']:
        path = root / base / label
        if (path / 'scores/result.json').exists():
            return path
    return None


def full_quality(path):
    if path is None:
        return None
    result = js(path / 'scores/result.json')
    if result.get('status') != 'complete':
        return None
    assert result['completed'] == result['expected'] == 103
    assert result['official_full_count'] and result['S_proxy_scope'] == 'full100_test_NR_plus3val_FR'
    assert result['test']['n'] == 100 and result['validation']['n'] == 3
    rows = [json.loads(x) for x in (path / 'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
    manifest = js(path / 'render_manifest.json'); rp = js(path / 'render_protocol.json')
    assert len(rows) == len(manifest['records']) == 103 and sum(len(r['metrics']) for r in rows) == 418
    assert [(r['split'], r['case']) for r in rows] == [('validation', i) for i in range(1, 4)] + [('test', i) for i in range(1, 101)]
    assert all(math.isfinite(v) for row in rows for v in row['metrics'].values())
    assert all(row['sr_sha256'] == rec['sha256'] for row, rec in zip(rows, manifest['records']))
    sp = js(path / 'scores/protocol.json')
    assert next(x['sha256'] for x in sp['original_sources'] if x['path'] == 'csig_metrics.py') == CM_SHA
    for split in ['validation', 'test']:
        group = [row for row in rows if row['split'] == split]
        for key in group[0]['metrics']:
            mean = sum(row['metrics'][key] for row in group) / len(group)
            assert abs(mean - result[split][key]) < 1e-12
    t, v = result['test'], result['validation']
    nr = t['clipiqa'] + t['maniqa'] + t['musiq'] / 100 + max(0, (10 - t['niqe']) / 10)
    fr = (1 - v['lpips']) + (1 - v['dists'])
    assert abs(nr - t['nr']) < 1e-12 and abs(fr - v['fr']) < 1e-12
    assert abs(t['nr'] + v['fr'] - result['S_proxy']) < 1e-12
    return dict(NR=t['nr'], FR=v['fr'], S=result['S_proxy'], render_protocol=rp['protocol'],
                result_sha256=sha(path / 'scores/result.json'), eval_path=str(path))


def join(root):
    source = root / 'scoring_tool/predict.py'
    if not source.exists():
        source = root / 'nr_push2_full_evaluations/audits/original_predict.py'
    assert sha(source) == PREDICT_SHA
    spec = importlib.util.spec_from_file_location('original_csig_predict', source)
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    reference = js(root / 'calibration/push2_g4_latency.json')
    ref_ms = reference['latency_512']['event_p10']
    calibrations = {k: js(root / f'calibration/push2_{k}_latency.json') for k in LABELS}
    qualities = {k: full_quality(locate(root, label)) for k, label in LABELS.items()}
    parent = qualities['parent6250']; assert parent is not None
    for key, calibration in calibrations.items():
        assert calibration['status'] == 'complete' and calibration['cuda'] == reference['cuda']
        assert calibration['input']['shape'] == [1, 3, 512, 512] and calibration['input']['dtype'] == 'torch.float16'
        assert calibration['ref_ms'] == ref_ms
        normalized = calibration['latency_512']['event_p10'] * original.ANCHOR['g'] / ref_ms
        assert abs(normalized - calibration['g_used_ms']) < 1e-12
        quality = qualities[key]
        if quality is None:
            continue
        p = quality['render_protocol']; runner = p['runner']
        assert calibration['model_dir'] == p['model_dir']
        manifest = calibration['artifacts']['files'] + calibration['artifacts']['extra_files']
        assert any(x['sha256'] == runner['checkpoint_sha256'] for x in manifest)
        assert any(x['path'] == 'runner.py' and x['sha256'] == runner['runner_sha256'] for x in manifest)
        assert any(x['path'] == 'config.json' and x['sha256'] == p['model_dir_sha256']['config.json'] for x in manifest)
    parent_g = calibrations['parent6250']['g_used_ms']
    parent_prediction = original.predict(parent['S'], parent_g)
    comparisons = {}
    for key, quality in qualities.items():
        if quality is None:
            comparisons[key] = dict(status='incomplete_or_absent', prediction=None)
            continue
        calibration = calibrations[key]; g = calibration['g_used_ms']
        quality_factor = 2 ** (quality['S'] - parent['S'])
        speed_factor = ((original.A + original.K * parent_g) / (original.A + original.K * g)) ** .2
        comparisons[key] = dict(status='complete103', NR=quality['NR'], FR=quality['FR'], S=quality['S'],
            measured_event_p10_ms=calibration['latency_512']['event_p10'], normalized_g_ms=g,
            measured_latency_prediction=original.predict(quality['S'], g),
            parent_latency_fixed_quality_only_prediction=original.predict(quality['S'], parent_g),
            delta_NR_vs_parent=quality['NR'] - parent['NR'], delta_FR_vs_parent=quality['FR'] - parent['FR'],
            delta_S_vs_parent=quality['S'] - parent['S'], quality_factor_vs_parent=quality_factor,
            measured_latency_factor_vs_parent_not_proven_speedup=speed_factor,
            combined_factor_vs_parent=quality_factor * speed_factor,
            result_sha256=quality['result_sha256'], latency_json_sha256=sha(root / f'calibration/push2_{key}_latency.json'))
    # A has no fresh latency in this four-file calibration set: quality only.
    a = full_quality(locate(root, 'nr_stage2_cont_s6500_compiled_all'))
    a_quality = None if a is None else dict(NR=a['NR'], FR=a['FR'], S=a['S'],
        delta_NR_vs_parent=a['NR'] - parent['NR'], delta_FR_vs_parent=a['FR'] - parent['FR'],
        delta_S_vs_parent=a['S'] - parent['S'], quality_factor_vs_parent=2 ** (a['S'] - parent['S']),
        hypothetical_prediction_at_parent_latency=original.predict(a['S'], parent_g),
        measured_combined_prediction=None, reason='No fresh A latency provided; do not reuse a different checkpoint timing.')
    return dict(schema='fresh_paired_full103_prediction_v1', original_predict_sha256=PREDICT_SHA,
        G4_event_p10_ms=ref_ms, G4_anchor_ms=original.ANCHOR['g'], parent_current_paired_prediction=parent_prediction,
        comparisons=comparisons, full_A_quality_only=a_quality,
        interpretation='Original-tool empirical prediction, not measured leaderboard score. Single-run timing factors are kept separate from quality gain; no statistical speedup claim.',
        physical_card_evidence='Same GPU0 supplied by launch provenance; JSON records logical0 and identical hardware/runtime, without physical UUID.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(); report = join(args.root)
    out = args.out or args.root / 'nr_push2_full_evaluations/audits/paired_full_prediction.json'
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
