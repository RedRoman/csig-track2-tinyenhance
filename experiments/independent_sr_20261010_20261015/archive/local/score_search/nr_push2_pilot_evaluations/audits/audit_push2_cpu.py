"""Read-only CPU audit of fixed23 pilot PNGs, metrics, provenance and RNG."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import base64
import datetime
import hashlib
import json
import math
import struct
import numpy as np
import torch
from PIL import Image

ROOT = Path('/cache/independent_sr_cuda_20261013')
NEW = ROOT / 'nr_push2_pilot_evaluations'
OLD = ROOT / 'nr_pilot_evaluations/nr_conservative_s6250_compiled_pilot'
LABELS = ['nr_s6250_ct175_freshcontrol_pilot', 'nr_s6250_ct160_compiled_pilot',
          'nr_s6250_ct185_compiled_pilot', 'nr_stage2_cont_s6350_compiled_pilot',
          'nr_stage2_cont_s6500_compiled_pilot', 'nr_stage2_roundste_s6500_compiled_pilot']
CASES = [1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100]
NR = ['niqe', 'musiq', 'maniqa', 'clipiqa']
VK = ['lpips', 'dists'] + NR
CACHE = {}


def sha(path):
    path = Path(path).resolve()
    if path not in CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(b)
        CACHE[path] = h.hexdigest()
    return CACHE[path]


def js(path):
    return json.loads(Path(path).read_text())


def norm_rng(value):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return ['tensor', str(value.dtype), list(value.shape), hashlib.sha256(array.tobytes()).hexdigest()]
    if isinstance(value, np.ndarray):
        return ['array', str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, (list, tuple)):
        return [norm_rng(x) for x in value]
    if isinstance(value, dict):
        return {k: norm_rng(v) for k, v in sorted(value.items())}
    if isinstance(value, float):
        return value.hex()
    return value


def audit(directory):
    res = js(directory / 'scores/result.json')
    rp = js(directory / 'render_protocol.json'); p = rp['protocol']
    sp = js(directory / 'scores/protocol.json')
    rows = [json.loads(x) for x in (directory / 'scores/measurements.jsonl').read_text().splitlines() if x.strip()]
    manifest = js(directory / 'render_manifest.json')
    assert res['status'] == 'complete' and res['completed'] == res['expected'] == 23
    assert not res['official_full_count'] and res['S_proxy_scope'] == 'partial_or_preflight_only'
    assert len(rows) == len(manifest['records']) == len(p['tasks']) == 23
    assert sum(len(x['metrics']) for x in rows) == 98
    assert [(x['split'], x['case']) for x in rows] == [('validation', i) for i in range(1, 4)] + [('test', i) for i in CASES]
    assert hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() == rp['protocol_id'] == manifest['protocol_id']
    for row, rec, inp in zip(rows, manifest['records'], p['tasks']):
        assert list(row['metrics']) == (VK if row['split'] == 'validation' else NR)
        assert all(math.isfinite(v) for v in row['metrics'].values())
        assert (row['split'].replace('validation', 'val'), row['case']) == (rec['split'], rec['case']) == (inp['split'], inp['case'])
        assert row['sr_sha256'] == rec['sha256'] == sha(directory / rec['output'])
        assert row['size_hw'] == rec['size'][::-1] == inp['size'][::-1]
        assert sha(inp['input']) == inp['input_sha256']
        if row['split'] == 'validation':
            gt = ROOT / 'scoring_tool/data/验证集' / ('case3_gt.jpg' if row['case'] == 3 else f'case{row["case"]}_gt.png')
            assert row['gt_sha256'] == sha(gt)
    means = {s: {k: sum(x['metrics'][k] for x in rows if x['split'] == s) / sum(x['split'] == s for x in rows)
                 for k in keys} for s, keys in [('test', NR), ('validation', VK)]}
    nr = lambda x: x['clipiqa'] + x['maniqa'] + x['musiq'] / 100 + max(0, (10 - x['niqe']) / 10)
    N = nr(means['test']); F = 2 - means['validation']['lpips'] - means['validation']['dists']; S = N + F
    deltas = [abs(res[s][k] - v) for s, values in means.items() for k, v in values.items()]
    deltas += [abs(res['test']['nr'] - N), abs(res['validation']['fr'] - F), abs(res['S_proxy'] - S)]
    assert max(deltas) < 1e-12
    r = p['runner']
    assert sha(r['weight_path']) == r['checkpoint_sha256']
    assert sha(ROOT / 'code/render_cuda_submission.py') == p['source_sha256']
    assert sha(Path(p['model_dir']) / 'runner.py') == r['runner_sha256']
    for rel, digest in r['source_sha256'].items():
        assert sha(Path(r['repo_path']) / rel) == digest
    model_files = {rel: sha(Path(p['model_dir']) / rel) == digest for rel, digest in p['model_dir_sha256'].items()}
    assert all(ok for rel, ok in model_files.items() if not rel.endswith('.pyc'))
    assert sha(ROOT / 'code/score_cuda_stream.py') == sp['script_sha256']
    for source in sp['original_sources']:
        assert sha(source['resolved_path']) == source['sha256']
    for key, path_key, hash_key in [('numpy_legacy_compat', 'path', 'sha256'),
                                  ('maniqa_offline_init_proof', 'adapter_path', 'adapter_sha256')]:
        assert sha(sp[key][path_key]) == sp[key][hash_key]
    mp = sp['maniqa_offline_init_proof']
    assert sha(mp['checkpoint_path']) == mp['checkpoint_sha256']
    assert mp['all_final_parameters_and_buffers_equal_checkpoint'] and mp['test_sample'] == 100
    summary = dict(label=directory.name, image_rows=23, scalar_metrics=98, NR=N, FR=F, S_subset=S,
                   max_formula_delta=max(deltas), component_means=means,
                   checkpoint_sha256=r['checkpoint_sha256'], coeff_t=r['coeff_t'],
                   model_dir_file_hash_matches=model_files, actual_png_input_GT_source_checkpoint_hashes='passed')
    return summary, rows, manifest, p, sp


def main():
    os.nice(10); torch.set_num_threads(2)
    old = audit(OLD); current = [audit(NEW / name) for name in LABELS]
    control = current[0]
    SK = ['seed', 'seed_scope', 'metric_order', 'environment_versions', 'environment_protocol_sha256',
          'original_tool_deployment_sha256', 'original_sources', 'script_sha256', 'image_policy',
          'numpy_legacy_compat', 'maniqa_offline_init_proof']
    assert all(all(x[4][k] == control[4][k] for k in SK) for x in [old] + current)
    assert all(x[3]['tasks'] == control[3]['tasks'] for x in [old] + current)
    comparisons = []; scalars_equal = 0
    for a, b, ma, mb in zip(old[1], control[1], old[2]['records'], control[2]['records']):
        pa, pb = OLD / ma['output'], NEW / LABELS[0] / mb['output']
        with Image.open(pa) as im: xa = np.asarray(im.convert('RGB')).copy()
        with Image.open(pb) as im: xb = np.asarray(im.convert('RGB')).copy()
        assert xa.shape == xb.shape
        changed = xa != xb
        values = {k: struct.pack('<d', a['metrics'][k]) == struct.pack('<d', b['metrics'][k]) for k in a['metrics']}
        scalars_equal += sum(values.values())
        comparisons.append(dict(split=a['split'], case=a['case'], old_sha256=sha(pa), fresh_sha256=sha(pb),
            byte_exact=sha(pa) == sha(pb), pixels_exact=not bool(np.any(changed)),
            changed_channel_elements=int(np.count_nonzero(changed)), metric_values_bitexact=values))
    rngs = [torch.load(path / 'render_progress.pt', map_location='cpu', weights_only=False) for path in (OLD, NEW / LABELS[0])]
    rng_report = {}
    for key in ['python_rng', 'numpy_rng', 'torch_rng', 'device_rng']:
        values = [norm_rng(x[key]) for x in rngs]
        digests = [hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for v in values]
        rng_report[key] = dict(equal=values[0] == values[1], old_sha256=digests[0], fresh_sha256=digests[1])
    report = dict(status='passed', audited_bjt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        cpu_only=True, cuda_initialized=torch.cuda.is_initialized(), no_rerender_or_source_changes=True,
        scope='fixed20 test +3validation pilot; S_subset, not full100 NR or official prediction',
        results=[x[0] for x in current], old_control=old[0],
        fresh_control_comparison=dict(pngs=comparisons, equal_scalar_metrics=scalars_equal, total_scalar_metrics=98,
            all23_png_byte_exact=all(x['byte_exact'] for x in comparisons),
            all23_png_pixels_exact=all(x['pixels_exact'] for x in comparisons), render_rng=rng_report,
            all4_render_rng_exact=all(x['equal'] for x in rng_report.values()),
            aggregate_result_component_values_exact=old[0]['component_means'] == control[0]['component_means']),
        all_candidate_input_GT_metric_protocol_matched=True,
        formula='mean components first; NR=C+M+U/100+max(0,(10-N)/10); FR=2-L-D; S_subset=NR_test20+FR_val3')
    assert not report['cuda_initialized']
    files = {}
    for directory in [OLD] + [NEW / n for n in LABELS]:
        for rel in ['render_protocol.json', 'render_manifest.json', 'render_status.json', 'render_progress.pt',
                    'scores/status.json', 'scores/protocol.json', 'scores/result.json',
                    'scores/measurements.jsonl', 'scores/rng_checkpoint.pt']:
            data = (directory / rel).read_bytes()
            files[directory.name + '/' + rel] = dict(base64=base64.b64encode(data).decode(), sha256=hashlib.sha256(data).hexdigest())
    print(json.dumps(dict(audit=report, files=files), ensure_ascii=False))


if __name__ == '__main__':
    main()
