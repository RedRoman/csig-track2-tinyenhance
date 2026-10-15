"""CPU-only formal HYPIR same-weight PNG103 comparison; no diagnostic PNG reuse.

Default prints a plan. --execute reads complete original-renderer cohorts and
writes one NEW proof file. It never imports a Runner or initializes CUDA.
Exact PNG103 plus matching formal inputs/protocol/final4RNG permits explicitly
referencing the frozen parent's already-audited metric result, not a new score.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

R = Path('/cache/independent_sr_cuda_20261013')
PARENT = R / 'nr_push4_full_evaluations/push4_a7750_compiled_all'
SELECTION = R / 'results/nr_push4_selection.json'
SELECTION_SHA = 'b3148b21f466854f9fe8f81c7eca1348da5bf793141799573ff3be8b20e25c3f'
RAW_SHA = '589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d'
RUNNER_SHA = '396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae'
RENDER_SHA = '1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5'
SCORER_SHA = '20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074'
CM_SHA = '1af7fecd0aa79ec9e8b9f914332cd414b9227338159813b7f45125676c8ba1f2'
MANIFEST_SHA = '3ada479b6425f1d2ff11e91250101efdd1ba6127e17072b9c7cd2b49307af1fa'
VAL = R / 'source_payload/scoring_assets/data/验证集'
TEST = R / 'source_payload/assets/input'
ORDER = [('val', i) for i in range(1, 4)] + [('test', i) for i in range(1, 101)]
OBSERVED = {}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    stamp = lambda s: (s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev)
    require(stamp(before) == stamp(after), f'file changed while reading: {path}')
    require(str(path) not in OBSERVED or OBSERVED[str(path)] == stamp(after), f'file changed across reads: {path}')
    OBSERVED[str(path)] = stamp(after)
    return h.hexdigest()


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def rng_hash(value):
    # Match the frozen independent audit's normalized JSON convention.
    import numpy as np
    import torch
    def normalized(x):
        if isinstance(x, torch.Tensor):
            return dict(dtype=str(x.dtype), shape=list(x.shape),
                        sha256=hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest())
        if isinstance(x, np.ndarray):
            return dict(dtype=str(x.dtype), shape=list(x.shape), sha256=hashlib.sha256(x.tobytes()).hexdigest())
        if isinstance(x, dict):
            return {k: normalized(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [normalized(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True).encode()).hexdigest()


def identity(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def cohort(directory, expected_model_dir, expected_runner_sha, expected_config_sha):
    from PIL import Image
    require(directory.is_dir(), f'cohort absent: {directory}')
    require((directory / 'render_complete').is_file(), f'no render_complete: {directory}')
    status = read(directory / 'render_status.json')
    require(status['status'] == 'complete' and status['completed'] == status['total'] == 103,
            f'not complete103: {directory}')
    manifest = read(directory / 'render_manifest.json')
    protocol = manifest['protocol']
    require(manifest['scope'] == 'test100+validation3', 'wrong formal scope')
    require(manifest['protocol_id'] == identity(protocol), 'manifest protocol identity mismatch')
    protocol_file = read(directory / 'render_protocol.json')
    require(protocol_file['protocol_id'] == manifest['protocol_id'] and protocol_file['protocol'] == protocol,
            'render protocol/manifest mismatch')
    require(protocol['schema'] == 'cuda_submission_quality_v1' and protocol['cases'] == 'all', 'wrong render schema/scope')
    require(protocol['source_sha256'] == RENDER_SHA, 'not frozen original renderer')
    require(protocol['warmup'] is False, 'formal render must have no latency warmup')
    require(protocol['inference_rng'] == 'seed231 before model initialization; continuous ordered val3 then tests',
            'wrong render231 continuous order')
    require(Path(protocol['model_dir']).resolve() == expected_model_dir, 'modeldir binding mismatch')
    require(protocol['runner']['runner_sha256'] == expected_runner_sha, 'runner source mismatch')
    require(protocol['model_dir_sha256']['runner.py'] == expected_runner_sha and
            protocol['model_dir_sha256']['config.json'] == expected_config_sha, 'modeldir captured source mismatch')
    require(sha(expected_model_dir / 'runner.py') == expected_runner_sha and
            sha(expected_model_dir / 'config.json') == expected_config_sha, 'current runner/config changed')
    require(protocol['runner']['checkpoint_sha256'] == RAW_SHA, 'not A7750 raw')
    tasks, records = protocol['tasks'], manifest['records']
    require(len(tasks) == len(records) == 103, 'tasks/records not103')
    pngs = set()
    for index, ((split, case), task, row) in enumerate(zip(ORDER, tasks, records)):
        src = VAL / f'case{case}_lq.jpg' if split == 'val' else TEST / f'case{case}.jpg'
        relative = f'render/{split}/case{case}.png'
        require((task['split'], task['case']) == (split, case), 'task order changed')
        require(Path(task['input']).resolve() == src.resolve(), 'not canonical formal input')
        require(sha(src) == task['input_sha256'], f'input JPEG changed: {src}')
        with Image.open(src) as im:
            require(list(im.size) == task['size'], 'source size mismatch')
        require((row['task_index'], row['split'], row['case'], row['output'], row['size']) ==
                (index, split, case, relative, task['size']), 'manifest row order/shape/path changed')
        dst = directory / relative
        require(dst.is_file() and not dst.is_symlink(), 'PNG missing/symlink')
        require(sha(dst) == row['sha256'], f'actual PNG differs from committed hash: {dst}')
        with Image.open(dst) as im:
            require(im.format == 'PNG' and im.mode == 'RGB' and list(im.size) == row['size'], 'bad canonical PNG format/size')
        pngs.add(relative)
    actual = {str(p.relative_to(directory)) for p in (directory / 'render').rglob('*') if p.is_file()}
    require(actual == pngs, f'noncanonical/missing/extra render files: {actual ^ pngs}')
    return manifest


def final_rng(directory, manifest):
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA must not initialize')
    sha(directory / 'render_progress.pt')
    saved = torch.load(directory / 'render_progress.pt', map_location='cpu', weights_only=False)
    require(saved['protocol_id'] == manifest['protocol_id'], 'state protocol mismatch')
    require(saved['records'] == manifest['records'], 'final state != committed103 manifest')
    require(saved['checkpoint_sha256'] == RAW_SHA and saved['input_sha256'] ==
            [x['input_sha256'] for x in manifest['protocol']['tasks']], 'state source/checkpoint mismatch')
    hashes = {k: rng_hash(saved[k]) for k in ['python_rng', 'numpy_rng', 'torch_rng', 'device_rng']}
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    return hashes


def parent_quality(selection, parent_manifest):
    audit_path = Path(selection['full_audit'])
    require(sha(audit_path) == selection['full_audit_sha256'], 'frozen parent audit changed')
    audit = read(audit_path)
    require(audit['status'] == 'PASS' and audit['image_rows'] == 103 and audit['scalar_metrics'] == 418,
            'parent not independently PASS103/418')
    require(Path(audit['evaluation_dir']) == PARENT and audit['actual_checkpoint_sha256'] == RAW_SHA,
            'parent audit binding mismatch')
    require(audit['render_protocol_id'] == parent_manifest['protocol_id'], 'parent audit/render identity changed')
    protocol_path = PARENT / 'scores/protocol.json'
    protocol = read(protocol_path)
    require(protocol['seed'] == 1234 and protocol['script_sha256'] == SCORER_SHA, 'wrong formal scoring protocol1234')
    require(sha(R / 'code/score_cuda_stream.py') == SCORER_SHA and
            sha(R / 'scoring_tool/csig_metrics.py') == CM_SHA, 'original scoring source changed')
    rows_path = PARENT / 'scores/measurements.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    require(len(rows) == len(audit['records']) == 103, 'parent score/audit records not103')
    count = 0
    for row, (split, case), image, task, frozen in zip(rows, ORDER, parent_manifest['records'],
                                                     parent_manifest['protocol']['tasks'], audit['records']):
        require((row['split'], row['case']) == ('validation' if split == 'val' else split, case), 'score order mismatch')
        require((frozen['split'], frozen['case']) == (row['split'], case), 'frozen parent audit row order changed')
        require(frozen['actual_png_sha256'] == image['sha256'] and
                frozen['actual_input_sha256'] == task['input_sha256'], 'current PNG/input not frozen audited parent')
        require(row['sr_sha256'] == image['sha256'] and Path(row['sr_path']) == PARENT / image['output'],
                'scored actual PNG mismatch')
        require(row['seed_scope'] == 'split_seed1234_continuous_rng', 'score RNG scope changed')
        expected = {'niqe', 'musiq', 'maniqa', 'clipiqa'} | ({'lpips', 'dists'} if split == 'val' else set())
        require(set(row['metrics']) == expected and all(math.isfinite(v) for v in row['metrics'].values()),
                'wrong metric records')
        count += len(row['metrics'])
        if split == 'val':
            gt = VAL / (f'case{case}_gt.jpg' if case == 3 else f'case{case}_gt.png')
            require(sha(gt) == row['gt_sha256'] == frozen['actual_gt_sha256'], 'actual validation GT changed')
        else:
            require(row['gt_sha256'] is None and frozen['actual_gt_sha256'] is None, 'test must have no GT')
    require(count == 418, 'not418 scalar measurements')
    test = rows[3:]
    avg = lambda name: statistics.fmean(r['metrics'][name] for r in test)
    nr = avg('clipiqa') + avg('maniqa') + avg('musiq') / 100 + max(0, (10 - avg('niqe')) / 10)
    fr = 2 - statistics.fmean(r['metrics']['lpips'] for r in rows[:3]) - statistics.fmean(r['metrics']['dists'] for r in rows[:3])
    result = read(PARENT / 'scores/result.json')
    require(result == audit['published_result'], 'parent result differs from frozen independent audit')
    for split, subset in [('validation', rows[:3]), ('test', rows[3:])]:
        for name, mean in audit['component_means'][split].items():
            require(abs(statistics.fmean(r['metrics'][name] for r in subset) - mean) <= 1e-12,
                    f'parent component mean differs from frozen independent audit: {split}/{name}')
    require(result['status'] == 'complete' and result['completed'] == result['expected'] == 103,
            'parent quality result incomplete')
    for name, actual in [('NR', nr), ('FR', fr), ('S_proxy', nr + fr)]:
        require(abs(actual - selection[name]) <= 1e-12 and abs(actual - audit[name]) <= 1e-12,
                f'parent formula/audit/selection mismatch: {name}')
    require(abs(result['S_proxy'] - nr - fr) <= 1e-12, 'parent result formula mismatch')
    return dict(NR=nr, FR=fr, S_proxy=nr + fr, image_rows=103, scalar_metrics=418,
                full_audit=str(audit_path), full_audit_sha256=sha(audit_path),
                score_records_sha256=sha(rows_path), score_protocol_sha256=sha(protocol_path),
                score_result_sha256=sha(PARENT / 'scores/result.json'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-dir', type=Path, default=R / 'nr_push6_full_evaluations/push6_a7750_ro_wavelet_all')
    ap.add_argument('--candidate-manifest', type=Path, default=R / 'submissions/hypir_push6_a7750_ro_wavelet/candidate_manifest.json')
    ap.add_argument('--parent-dir', type=Path, default=PARENT)
    ap.add_argument('--out', type=Path, default=R / 'audits/push6/formal_png103/ro_wavelet_exact103_v2.json')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    a.candidate_dir, a.candidate_manifest, a.parent_dir, a.out = [p.resolve() for p in
        [a.candidate_dir, a.candidate_manifest, a.parent_dir, a.out]]
    require(a.parent_dir == PARENT, 'only frozen A7750 parent allowed')
    require(a.out.is_relative_to(R / 'audits/push6') and not a.out.exists(), 'output must be NEW under audits/push6')
    if not a.execute:
        print(json.dumps(dict(status='CPU_PLAN_ONLY', formal_count=103, render_seed=231, scorer_seed=1234,
              parent_dir=str(a.parent_dir), candidate_dir=str(a.candidate_dir), candidate_manifest=str(a.candidate_manifest),
              out=str(a.out), GPU_calls=False, scope='actual formal PNG103; never diagnostic PNG'), indent=2))
        return
    require(a.candidate_dir.is_relative_to(R / 'nr_push6_full_evaluations'), 'not a new formal Push6 directory')
    require(sha(SELECTION) == SELECTION_SHA, 'frozen parent selection changed')
    selection = read(SELECTION)
    require(Path(selection['quality_dir']) == PARENT and selection['checkpoint_sha256'] == RAW_SHA and
            selection['runner_source_sha256'] == RUNNER_SHA, 'parent selection binding changed')
    require(sha(a.candidate_manifest) == MANIFEST_SHA, 'candidate manifest is not sealed RO+wavelet source')
    candidate = read(a.candidate_manifest)
    md = Path(candidate['model_dir']).resolve()
    require(candidate['checkpoint_sha256'] == RAW_SHA and candidate['reference_runner_sha256'] == RUNNER_SHA and
            candidate['formal_parent_quality_dir'] == str(PARENT), 'candidate manifest parent/weight binding changed')
    require(candidate['channels_last_conversion'] is False, 'channels_last excluded for quality preserving route')
    require(sha(candidate['checkpoint']) == RAW_SHA, 'actual raw changed')
    require(sha(R / 'code/render_cuda_submission.py') == RENDER_SHA, 'original renderer changed')
    parent_md = Path(selection['model_dir'])
    parent_config = read(parent_md / 'config.json')
    config = read(md / 'config.json')
    require({k: v for k, v in parent_config.items() if k != 'label'} ==
            {k: v for k, v in config.items() if k != 'label'}, 'candidate config changed beyond label')
    parent = cohort(PARENT, parent_md, RUNNER_SHA, sha(parent_md / 'config.json'))
    new = cohort(a.candidate_dir, md, candidate['runner_sha256'], candidate['config_sha256'])
    pp, np = parent['protocol'], new['protocol']
    for key in ['schema', 'tasks', 'cases', 'input_encoding', 'output_encoding', 'warmup', 'inference_rng',
                'source_sha256', 'gpu', 'torch']:
        require(pp[key] == np[key], f'formal protocol differs: {key}')
    allowed = {'label', 'runner_path', 'runner_sha256', 'compilation', 'speed_candidate'}
    require({k: v for k, v in pp['runner'].items() if k not in allowed} ==
            {k: v for k, v in np['runner'].items() if k not in allowed}, 'nonbackend Runner protocol changed')
    for relative, digest in pp['runner']['source_sha256'].items():
        require(sha(Path(pp['runner']['repo_path']) / relative) == digest, 'HYPIR source changed')
    quality = parent_quality(selection, parent)
    rng_parent, rng_new = final_rng(PARENT, parent), final_rng(a.candidate_dir, new)
    require(rng_parent == read(selection['full_audit'])['render_post_rng_hashes'], 'parent RNG audit mismatch')
    rows = [dict(split=p['split'], case=p['case'], parent_sha256=p['sha256'], candidate_sha256=n['sha256'],
                 exact=p['sha256'] == n['sha256']) for p, n in zip(parent['records'], new['records'])]
    matches = sum(row['exact'] for row in rows)
    all_rng_equal = rng_parent == rng_new
    exact = matches == 103 and all_rng_equal
    status = 'PASS_EXACT103' if exact else 'PNG_CHANGED_REQUIRES_INDEPENDENT103_418' if matches != 103 else 'RNG_CHANGED_REQUIRES_REVIEW'
    proof = dict(schema='push6_formal_png103_comparison_v2', status=status, utc=datetime.now(timezone.utc).isoformat(),
        CPU_only=True, CUDA_initialized=False, parent_dir=str(PARENT), candidate_dir=str(a.candidate_dir),
        parent_selection_sha256=SELECTION_SHA, candidate_manifest=str(a.candidate_manifest),
        candidate_manifest_sha256=sha(a.candidate_manifest), helper_sha256=sha(__file__),
        checkpoint_sha256=RAW_SHA, parent_runner_sha256=RUNNER_SHA, candidate_runner_sha256=candidate['runner_sha256'],
        render_seed=231, scorer_seed=1234, formal_order='val1,2,3 then test1..100; continuous render RNG; no warmup',
        actual_formal_inputs_checked=103, actual_validation_GT_checked=3,
        actual_PNG_checked_each_cohort=103, extra_render_files=0,
        PNG_exact_count=matches, final_fourRNG_equal=all_rng_equal, parent_final_fourRNG=rng_parent,
        candidate_final_fourRNG=rng_new, parent_quality=quality, rows=rows,
        RNG_proof_scope='actual final committed4RNG only; not per-call4RNG for formal103',
        allowed_runner_metadata_differences=sorted(allowed),
        metric_input_reuse_eligible=exact, quality_result_source='frozen A7750 independently audited103/418; not a new metric run',
        candidate_quality_from_exact_input_equivalence=quality if exact else None,
        parent_render_manifest_sha256=sha(PARENT / 'render_manifest.json'),
        candidate_render_manifest_sha256=sha(a.candidate_dir / 'render_manifest.json'),
        no_latency_claim=True, no_official_prediction=True)
    for path, expected in OBSERVED.items():
        s = Path(path).stat()
        require((s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev) == expected, f'source changed during audit: {path}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(proof, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=status, PNG_exact_count=matches, final_fourRNG_equal=all_rng_equal,
                          proof=str(a.out), proof_sha256=sha(a.out))))


if __name__ == '__main__':
    main()
