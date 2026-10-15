"""Join complete audited NR fine-tuning qualities to fresh CUDA measurements."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
from join_quality_latency import quality, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    server = (root / 'evaluations').is_dir()
    predictor = root / 'scoring_tool/predict.py' if server else root.parent / 'predict.py'
    spec = importlib.util.spec_from_file_location('original_predict', predictor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref_path = root / ('calibration/g4_latency.json' if server else 'cuda30416_calibration/g4_latency.json')
    ref = json.loads(ref_path.read_text())
    ref_ms = ref['latency_512']['event_p10']
    assert ref['status'] == 'complete'
    rows = []
    mappings = [
        ('s6000_mt200_ct175_compiled_cuda_all', None, 0, 'hypir_s6000_ct175_cached_fused_compiled_latency.json'),
        ('nr_conservative_s6100_compiled_all', 'nr_full_evaluations' if server else 'nr_cuda_evaluations', 100, 'hypir_nr_conservative_s6100_latency.json'),
        ('nr_conservative_s6250_compiled_all', 'nr_full_evaluations' if server else 'nr_cuda_evaluations', 250, 'hypir_nr_conservative_s6250_latency.json'),
    ]
    for label, directory, steps, filename in mappings:
        row = quality(root, label, directory)
        latency_path = root / 'calibration' / filename if server else root / ('cuda_runner/compile_variant' if steps == 0 else 'nr_cuda_calibration') / filename
        latency = json.loads(latency_path.read_text())
        assert latency['status'] == 'complete'
        assert latency['latency_512']['warm'] == 30 and latency['latency_512']['n'] == 200
        assert latency['cuda']['name'] == ref['cuda']['name'] == 'NVIDIA A100-SXM4-80GB'
        assert math.isclose(latency['ref_ms'], ref_ms, abs_tol=1e-12)
        assert row['checkpoint_sha256'] in {r['sha256'] for r in latency['artifacts']['extra_files']}
        ms = latency['latency_512']['event_p10']
        g = ms * module.ANCHOR['g'] / ref_ms
        assert math.isclose(g, latency['g_used_ms'], abs_tol=1e-12)
        row.update(extra_training_steps=steps, event_p10_ms=ms, wall_median_ms=latency['latency_512']['wall_med'],
                   g_calibrated_ms=g, predicted_score=module.predict(row['S_proxy'], g),
                   latency_source=str(latency_path), latency_sha256=sha(latency_path), model_dir=latency['model_dir'])
        rows.append(row)
    baseline = rows[0]
    for row in rows:
        row['delta_NR'] = row['test_NR'] - baseline['test_NR']
        row['delta_S'] = row['S_proxy'] - baseline['S_proxy']
        row['delta_predicted_score'] = row['predicted_score'] - baseline['predicted_score']
    result = dict(status='complete', score_type='user original empirical predictor; not official leaderboard',
                  scope='Each row independently renders native test100+validation3 PNG and measures fresh warm30+n200 single A100 GPU0 infer.',
                  caveat='Prediction is extrapolated from five pixel-model fit points; no HYPIR +/-2 confidence guarantee. Small timing differences are measurements, not evidence of algorithmic speed gain.',
                  g4_event_p10_ms=ref_ms, g4_reference_sha256=sha(ref_path), predictor_sha256=sha(predictor),
                  selected_label=max(rows, key=lambda x: x['test_NR'])['label'], rows=rows)
    output = root / 'results/nr_full_quality_latency_comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps([{k:r[k] for k in ['extra_training_steps','test_NR','val_FR','S_proxy','event_p10_ms','predicted_score']} for r in rows], indent=2))


if __name__ == '__main__':
    main()
