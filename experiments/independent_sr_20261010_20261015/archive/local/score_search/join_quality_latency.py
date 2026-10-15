"""Join only audited full CUDA qualities to same-card measured latencies.

Uses the user's unmodified predict.py. NPU/pilot/partial quality is rejected.
The cache-only row explicitly reuses mathematically identical original
quality; fusion/compilation always requires its independent full103 render.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def quality(root, label, evaluation_dir=None):
    directory = root / (evaluation_dir or ('evaluations' if (root / 'evaluations').is_dir() else 'results')) / label
    path = directory / 'scores/result.json'
    d = json.loads(path.read_text())
    assert d['status'] == 'complete' and d['completed'] == d['expected'] == 103
    assert d['S_proxy_scope'] == 'full100_test_NR_plus3val_FR'
    rows_path = directory / 'scores/measurements.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    assert [(r['split'], r['case']) for r in rows] == [('validation', i) for i in range(1, 4)] + [('test', i) for i in range(1, 101)]
    assert sum(len(r['metrics']) for r in rows) == 418
    assert all(math.isfinite(v) for r in rows for v in r['metrics'].values())
    manifest_path = directory / 'render_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['records']) == 103
    for row, png in zip(rows, manifest['records']):
        assert row['case'] == png['case'] and row['sr_sha256'] == png['sha256']
        assert row['split'] == ('validation' if png['split'] == 'val' else png['split'])
        assert row['size_hw'] == list(reversed(png['size']))
    means = {split: {k: statistics.fmean(row['metrics'][k] for row in rows if row['split'] == split)
                     for k in (['lpips', 'dists', 'niqe', 'musiq', 'maniqa', 'clipiqa'] if split == 'validation' else ['niqe', 'musiq', 'maniqa', 'clipiqa'])}
             for split in ('validation', 'test')}
    nr = means['test']['clipiqa'] + means['test']['maniqa'] + means['test']['musiq'] / 100 + max(0, (10 - means['test']['niqe']) / 10)
    fr = 2 - means['validation']['lpips'] - means['validation']['dists']
    assert math.isclose(nr, d['test']['nr'], abs_tol=1e-12)
    assert math.isclose(fr, d['validation']['fr'], abs_tol=1e-12)
    assert math.isclose(nr + fr, d['S_proxy'], abs_tol=1e-12)
    return dict(label=label, test_NR=nr, val_FR=fr, S_proxy=nr + fr,
                checkpoint_sha256=manifest['protocol']['runner']['checkpoint_sha256'],
                quality_source_sha256=sha(path), measurements_sha256=sha(rows_path),
                render_manifest_sha256=sha(manifest_path),
                quality_source=str(path.resolve()), quality_protocol='native CUDA FP16 submission input; BF16 model; PNG round; test100+validation3')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    root = a.root.resolve()
    server_layout = (root / 'evaluations').is_dir()
    predictor_path = root / 'scoring_tool/predict.py' if server_layout else root.parent / 'predict.py'
    spec = importlib.util.spec_from_file_location('original_csig_predict', predictor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref_path = root / ('calibration/g4_latency.json' if server_layout else 'cuda30416_calibration/g4_latency.json')
    ref = json.loads(ref_path.read_text())
    ref_ms = ref['latency_512']['event_p10']
    mappings = [
        ('s8000_mt200_ct200_cuda_all', 'original', root / ('calibration/hypir_s8000_ct200_latency.json' if server_layout else 'cuda_runner/compile_variant/hypir_s8000_ct200_latency.json')),
        ('s6000_mt200_ct175_cuda_all', 'original', root / ('calibration/hypir_s6000_ct175_latency.json' if server_layout else 'cuda30416_calibration/hypir_s6000_ct175_latency.json')),
        ('s6000_mt200_ct175_cuda_all', 'cached_empty_caption', root / ('calibration/hypir_s6000_ct175_cached_latency.json' if server_layout else 'cuda_runner/cache_variants/hypir_s6000_ct175_cached_latency.json')),
        ('s6000_mt200_ct175_compiled_cuda_all', 'cached_fused_compiled_tiles', root / ('calibration/hypir_s6000_ct175_cached_fused_compiled_latency.json' if server_layout else 'cuda_runner/compile_variant/hypir_s6000_ct175_cached_fused_compiled_latency.json')),
    ]
    cache_proof_path = root / ('calibration/cache_variants_validation.json' if server_layout else 'cuda_runner/cache_variants/cache_variants_validation.json')
    cache_proof = json.loads(cache_proof_path.read_text())
    assert cache_proof['status'] == 'passed' and all(cache_proof['initial_rng_equal'].values())
    assert all(row['gpu_float_bitexact'] and row['uint8_round_equal'] and all(row['rng_equal'].values()) for row in cache_proof['checks'])
    rows = []
    for label, variant, latency_path in mappings:
        row = quality(root, label)
        latency = json.loads(latency_path.read_text())
        assert latency['status'] == 'complete'
        assert latency['latency_512']['warm'] == 30 and latency['latency_512']['n'] == 200
        assert latency['cuda']['name'] == ref['cuda']['name'] == 'NVIDIA A100-SXM4-80GB'
        assert math.isclose(latency['ref_ms'], ref_ms, abs_tol=1e-12)
        assert row['checkpoint_sha256'] in {r['sha256'] for r in latency['artifacts']['extra_files']}
        ms = latency['latency_512']['event_p10']
        g = ms * module.ANCHOR['g'] / ref_ms
        assert math.isclose(g, latency['g_used_ms'], abs_tol=1e-12)
        row.update(variant=variant, event_p10_ms=ms, wall_median_ms=latency['latency_512']['wall_med'],
                   g_calibrated_ms=g, predicted_score=module.predict(row['S_proxy'], g),
                   latency_source=str(latency_path.resolve()), latency_sha256=sha(latency_path),
                   server_model_dir=latency['model_dir'],
                   server_quality_source='/cache/independent_sr_cuda_20261013/evaluations/' + label + '/scores/result.json')
        if variant == 'cached_empty_caption':
            row.update(quality_reuse='fixed empty caption embedding/time cache; original arithmetic and all inference RNG unchanged',
                       cache_equivalence_proof_sha256=sha(cache_proof_path))
        rows.append(row)
    rows.sort(key=lambda x: x['predicted_score'], reverse=True)
    result = dict(status='complete', score_type='user-supplied empirical predictor, not official leaderboard',
                  prediction_scope='five-point pixel-model fit extrapolated to HYPIR; no claimed +/-2 confidence bound',
                  same_card_g4_event_p10_ms=ref_ms, g4_reference_sha256=sha(ref_path),
                  predictor_sha256=sha(predictor_path), rows=rows,
                  selected_variant=rows[0]['variant'], selected_quality_label=rows[0]['label'])
    out = root / 'results/quality_latency_comparison.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps([dict(label=r['label'], variant=r['variant'], S_proxy=r['S_proxy'],
                           event_p10_ms=r['event_p10_ms'], predicted_score=r['predicted_score']) for r in rows], indent=2))


if __name__ == '__main__':
    main()
