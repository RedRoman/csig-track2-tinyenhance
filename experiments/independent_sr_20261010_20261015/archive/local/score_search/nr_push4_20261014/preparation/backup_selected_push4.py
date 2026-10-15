"""Persist final Push4 exact resume + raw ZIP byte-recipe + optional previews.

Default is dry-run. Standard library only: ZIP metadata uses an inert whitelisted
unpickler, never torch/NumPy or CUDA. Direct byte writes and full SHA256 readback;
no chmod, copy2, SFS rename, deletion, or writes to previous backup prefixes.
"""
from __future__ import annotations
import argparse
import codecs
import collections
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import struct
import zipfile

ROOT = Path('/cache/independent_sr_cuda_20261013')
DEST = Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014')
CHUNK = 8 * 1024**2
SMALL = 16 * 1024**2
ALLOWANCE = 16 * 1024**2
RECIPE_SOURCE_SHA = '43b135a68047a9319e355d9967b4c1bcddcab16ff58d775a8c8f7ef261d932ce'
FROZEN_TOKENS = {}
PRIOR = [
    {'directory': '/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014',
     'marker': 'DONE',
     'manifest_sha256': 'ef284432a338b0a4574f52be645f4a40571ef94e775e1e39a2c0d1cfc1d13e8d',
     'DONE_sha256': '0a9cb512123ca402baaf43e21db2204853b2033fbcb61a1739f802306b36703f'},
    {'directory': '/temp/independent_sr_cuda_20261013_artifacts/nr_microtune_20261014',
     'marker': 'DONE.json',
     'manifest_sha256': '99f461e6e95045cf0af053dc3b24bde74cd3531715225d0ccf95eef2d2dbd0d4',
     'DONE_sha256': '92cadf2e81aba1864e027e1fded512c008b15585018e1c1eda7d9658b45b8d12'},
    {'directory': '/temp/independent_sr_cuda_20261013_artifacts/nr_push3_20261014',
     'marker': 'DONE',
     'manifest_sha256': '6c4cc6db3d963762cb1868d83d89969083326cc0e3932d64743a23b598c48eaa',
     'DONE_sha256': '15846dd6f53e3327bf7331d00a27d4928b2078edca4244ee29186623cf85655d'},
]


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


require = need


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def token(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frozen_file(path, expected):
    p = Path(path).resolve(); before = token(p)
    need(sha(p) == expected and token(p) == before, f'Frozen source changed: {p}')
    FROZEN_TOKENS[str(p)] = before
    return p


def under_root(path):
    p = Path(path)
    need(p.is_absolute(), f'Absolute source path required: {p}')
    need(not p.is_symlink(), f'Symlink source forbidden: {p}')
    p = p.resolve()
    need(p.is_relative_to(ROOT), f'Source outside experiment root: {p}')
    need(p.is_file(), f'Missing required source file: {p}')
    return p


def safe_target(relative):
    r = Path(relative)
    need(not r.is_absolute() and '..' not in r.parts, f'Unsafe backup path: {r}')
    p = DEST / r
    for q in [DEST, p, *p.parents]:
        if q == DEST.parent:
            break
        need(not q.is_symlink(), f'Symlink destination forbidden: {q}')
    return p


def previous_heads():
    records = []
    for row in PRIOR:
        p = Path(row['directory'])
        need(p != DEST, 'Previous backup prefix cannot equal destination')
        mb = (p / 'manifest.json').read_bytes()
        db = (p / row['marker']).read_bytes()
        md, dd = hashlib.sha256(mb).hexdigest(), hashlib.sha256(db).hexdigest()
        need((md, dd) == (row['manifest_sha256'], row['DONE_sha256']), f'Prior backup head changed: {p}')
        manifest, done = json.loads(mb), json.loads(db)
        need(done.get('status') == 'complete' and done.get('manifest_sha256') == md,
             f'Prior manifest/DONE checksum mismatch: {p}')
        need(done.get('destination') == str(p), f'Prior DONE destination mismatch: {p}')
        need(manifest.get('destination') == str(p), f'Prior manifest destination mismatch: {p}')
        need(len(manifest['files']) == len({x['path'] for x in manifest['files']}), 'Prior duplicate manifest records')
        records.append(dict(row, bytes=manifest['bytes'], count=len(manifest['files'])))
    return records


class InertNumpyObject:
    def __init__(self, *args):
        self.args = args
    def __setstate__(self, state):
        self.state = state


class StorageReference:
    def __init__(self, kind, key, location, count):
        self.kind, self.key, self.location, self.count = kind, key, location, count


class TensorReference:
    def __init__(self, storage, offset, size, stride, *unused):
        self.storage, self.offset, self.size, self.stride = storage, offset, size, stride


class MetadataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        pair = module, name
        if pair == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if pair == ('torch._utils', '_rebuild_tensor_v2'):
            return TensorReference
        if module == 'torch' and name in ('FloatStorage', 'ByteStorage'):
            return name
        if pair == ('_codecs', 'encode'):
            return codecs.encode
        if pair in (('numpy', 'dtype'), ('numpy', 'ndarray'), ('numpy._core.multiarray', '_reconstruct')):
            return InertNumpyObject
        raise ValueError(f'Unsupported checkpoint pickle global: {module}.{name}')

    def persistent_load(self, value):
        need(isinstance(value, tuple) and len(value) == 5 and value[0] == 'storage', 'Unsupported storage reference')
        _, kind, key, location, count = value
        need(kind in ('FloatStorage', 'ByteStorage'), 'Unsupported tensor storage dtype')
        need(isinstance(key, str) and key.isdigit(), 'Invalid storage key')
        need(isinstance(location, str) and (location == 'cpu' or location.startswith('cuda:')), 'Unknown storage location')
        need(isinstance(count, int) and count >= 0, 'Invalid storage element count')
        return StorageReference(kind, key, location, count)


def archive_metadata(archive):
    names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
    need(len(names) == 1, 'Ambiguous checkpoint data.pkl')
    name = names[0]
    need(archive.getinfo(name).file_size <= 2 * 1024**2, 'Unexpectedly large checkpoint metadata')
    return MetadataOnlyUnpickler(io.BytesIO(archive.read(name))).load(), name[:-len('data.pkl')]


def storage_digest(archive, prefix, tensor, dtype='FloatStorage'):
    need(isinstance(tensor, TensorReference) and tensor.storage.kind == dtype, 'Tensor dtype mismatch')
    need(tensor.offset == 0 and math.prod(tensor.size) == tensor.storage.count, 'Only complete dense tensor storages supported')
    name = prefix + 'data/' + tensor.storage.key
    width = 4 if dtype == 'FloatStorage' else 1
    need(archive.getinfo(name).file_size == tensor.storage.count * width, 'Tensor storage size mismatch')
    h = hashlib.sha256()
    with archive.open(name) as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_proof(weight, resume, cfg, selection):
    expected = {k: selection[k] for k in ('local_step', 'global_step', 'optimizer_step')}
    with zipfile.ZipFile(weight) as wa, zipfile.ZipFile(resume) as ra:
        raw, wp = archive_metadata(wa)
        state, rp = archive_metadata(ra)
        need(all(state.get(k) == v for k, v in expected.items()), 'Selected resume step counters differ')
        need(state.get('protocol_id') == cfg['protocol_id'], 'Selected resume protocol mismatch')
        need(state.get('parent_lineage') == cfg['protocol']['parent_lineage'], 'Selected resume parent lineage mismatch')
        lora = state['lora']
        need(len(raw) == len(lora) == 514 and list(raw) == list(lora), 'Raw/resume LoRA514 order mismatch')
        elements = 0
        for name, tensor in lora.items():
            source = raw[name]
            need(isinstance(tensor, TensorReference) and isinstance(source, TensorReference), 'Expected tensor references')
            need((tensor.size, tensor.stride) == (source.size, source.stride), f'Raw/resume shape/stride mismatch: {name}')
            need(storage_digest(wa, wp, source) == storage_digest(ra, rp, tensor), f'Raw/resume exact FP32 storage differs: {name}')
            elements += math.prod(tensor.size)
        need(elements == 259474432, 'LoRA element count changed')
        opt = state['optimizer']; ids = [pid for g in opt['param_groups'] for pid in g['params']]
        need(len(ids) == len(set(ids)) == len(opt['state']) == 514 and set(ids) == set(opt['state']), 'Adam IDs incomplete')
        endian = '<'
        if rp + 'byteorder' in ra.namelist():
            order = ra.read(rp + 'byteorder').decode().strip()
            need(order in ('little', 'big'), 'Unknown checkpoint byte order')
            endian = '<' if order == 'little' else '>'
        for pid, tensor in zip(ids, lora.values()):
            s = opt['state'][pid]
            for key in ('exp_avg', 'exp_avg_sq'):
                v = s[key]
                need(isinstance(v, TensorReference) and v.storage.kind == 'FloatStorage' and v.size == tensor.size,
                     f'Adam FP32 moment shape/dtype mismatch: {pid}/{key}')
            step = s['step']
            need(isinstance(step, TensorReference) and step.size == () and step.storage.kind == 'FloatStorage', 'Adam step must be FP32 scalar')
            with ra.open(rp + 'data/' + step.storage.key) as stream:
                stream.read(step.offset * 4)
                value = struct.unpack(endian + 'f', stream.read(4))[0]
            need(value == selection['optimizer_step'], 'Adam tensor step differs from selected milestone')
        need(all(k in state for k in ('python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng')), 'Missing exact-restart RNG states')
        need(isinstance(state['python_rng'], tuple) and len(state['python_rng']) == 3, 'Python RNG schema differs')
        need(isinstance(state['numpy_rng'], tuple) and len(state['numpy_rng']) == 5, 'NumPy RNG schema differs')
        for key in ('torch_rng', 'cuda_rng'):
            need(isinstance(state[key], TensorReference) and state[key].storage.kind == 'ByteStorage', f'{key} is not byte tensor')
    return dict(expected, lora_fp32_tensors=514, lora_elements=elements, raw_resume_all514_exact_FP32=True,
                adam_states=514, adam_moments_FP32_shape_verified=True, all_Adam_steps=selection['optimizer_step'],
                all_four_rng_states_present=True, method='stdlib ZIP + inert whitelisted pickle + all514 actual LoRA storage SHA; no tensor libraries')


def exact_raw_recipe_proof(selection, resume):
    directory = Path(selection['raw_recipe_dir']).resolve()
    need(directory.is_relative_to(ROOT / 'audits/push4') and directory.is_dir(), 'Selected raw recipe must live under new audits/push4')
    recipe_path = under_root(directory / 'recipe.json'); frozen_file(recipe_path, sha(recipe_path)); recipe = read(recipe_path)
    need(recipe['schema'] == 'raw_torch_zip_exact_byte_recipe_v1' and recipe['tensor_entries'] == 514,
         'Raw ZIP exact-byte recipe schema/count differs')
    raw = Path(selection['checkpoint'])
    need(recipe['raw_name'] == raw.name and recipe['raw_bytes'] == raw.stat().st_size
         and recipe['raw_sha256'] == selection['checkpoint_sha256'], 'Recipe raw identity differs from selected checkpoint')
    need(recipe['resume_name'] == resume.name and recipe['resume_bytes'] == resume.stat().st_size
         and recipe['resume_sha256'] == selection['resume_sha256'], 'Recipe does not reference selected immutable resume')
    gaps = under_root(directory / recipe['gaps_file'])
    need(gaps.parent == directory and gaps.stat().st_size <= SMALL, 'Unexpected large/escaped raw ZIP gaps file')
    frozen_file(gaps, recipe['gaps_sha256'])
    helper = under_root(ROOT / 'code/push4/checkpoint_archive_recipe.py'); frozen_file(helper, RECIPE_SOURCE_SHA)
    proof_path = under_root(selection['raw_recipe_restore_proof'])
    frozen_file(proof_path, selection['raw_recipe_restore_proof_sha256']); proof = read(proof_path)
    need(proof['status'] == 'PASS' and proof['CPU_only'] is True
         and proof['raw_restore_complete_byte_SHA256'] == selection['checkpoint_sha256']
         and proof['resume_sha256'] == selection['resume_sha256'] and proof['helper_sha256'] == RECIPE_SOURCE_SHA,
         'Selected recipe requires its own successful complete-byte restore proof; a parent probe is insufficient')
    # Independently reconstruct the entire original raw byte hash in memory-sized
    # chunks. This writes no redundant raw file on either /cache or SFS.
    total = 0; entries = 0; h = hashlib.sha256()
    with zipfile.ZipFile(resume) as archive, gaps.open('rb') as gap_stream:
        for segment in recipe['segments']:
            count = segment['bytes'];need(isinstance(count, int) and count >= 0, 'Invalid recipe segment size')
            if segment['kind'] == 'gap':
                need(count <= SMALL, 'Unexpected oversized literal gap')
                gap_stream.seek(segment['offset']);data = gap_stream.read(count)
                need(len(data) == count, 'Incomplete literal gap');h.update(data);total += count
            else:
                need(segment['kind'] == 'resume_entry', 'Unknown recipe segment kind')
                info = archive.getinfo(segment['name']);need(info.file_size == count, 'Recipe ZIP entry size differs')
                eh = hashlib.sha256(); size = 0
                with archive.open(info) as source:
                    for block in iter(lambda: source.read(CHUNK), b''):h.update(block);eh.update(block);size += len(block)
                need(size == count and eh.hexdigest() == segment['sha256'], 'Recipe actual resume entry differs')
                total += size;entries += 1
    need(total == recipe['raw_bytes'] and h.hexdigest() == selection['checkpoint_sha256'] and entries == 514,
         'Resume plus gaps cannot reconstruct exact selected raw ZIP bytes')
    return {'status': 'PASS', 'raw_sha256': h.hexdigest(), 'raw_bytes': total, 'tensor_entries': entries,
            'recipe_sha256': sha(recipe_path), 'gaps_sha256': sha(gaps), 'recipe_helper_sha256': RECIPE_SOURCE_SHA,
            'raw_persisted_as_duplicate': False, 'exact_raw_reconstructable_from_persisted_resume_and_recipe': True,
            'method': 'streamed exact original raw ZIP hash from retained resume entries plus literal gaps; no output file'}, {recipe_path, gaps, helper, proof_path}


def audit_gates(selection):
    full_path = under_root(selection['full_audit'])
    train_path = under_root(selection['training_audit'])
    need(sha(full_path) == selection['full_audit_sha256'], 'Independent full-audit SHA differs')
    need(sha(train_path) == selection['training_audit_sha256'], 'Independent training-audit SHA differs')
    full, train = read(full_path), read(train_path)
    need(full.get('status') == 'PASS' and full.get('image_rows') == 103 and full.get('scalar_metrics') == 418,
         'Independent full audit must PASS103/418')
    need(full.get('label') == selection['label'] and full.get('actual_checkpoint_sha256') == selection['checkpoint_sha256'],
         'Full audit label/checkpoint differs')
    need(full.get('actual_png_manifest_scored_sha') == 'PASS103' and full.get('actual_source_code_sha') == 'PASS',
         'Full PNG/source audit is not PASS')
    for relative, record in full['actual_metric_weights_sha256'].items():
        frozen_file(ROOT / 'scoring_tool/weights' / relative, record['sha256'])
    need(train.get('status') == 'PASS' and train.get('CPU_only') is True and train.get('CUDA_context_initialized') is False,
         'Independent training audit must PASS without CUDA context')
    run = Path(selection['config']).parent.name
    matches = [row for row in train['branches'] if row['run'] == run and row['audited_global_step'] == selection['global_step']]
    need(len(matches) == 1, 'Training audit does not uniquely cover selected run/milestone')
    row = matches[0]
    need(row['status'] == 'PASS' and row['audited_stage_local_step'] == selection['local_step']
         and row['audited_Adam_step'] == selection['optimizer_step'], 'Training audit step mismatch')
    need(row['LoRA']['keys'] == 514 and row['LoRA']['dtype'] == 'torch.float32' and row['LoRA']['all_finite'] is True,
         'Independent audit lacks all514 finite FP32 LoRA proof')
    need(row['final_weight_exact_equal_resume_FP32_LoRA'] is True and row['Adam']['state_entries'] == 514
         and row['Adam']['all_moments_FP32_CPU_finite'] is True
         and row['Adam']['all_steps'] == [selection['optimizer_step']], 'Exact raw/resume/Adam proof absent')
    need(set(row['final_rng']['state_sha256']) == {'python', 'numpy', 'cpu', 'cuda'}, 'Independent final4 RNG proof absent')
    for field in ('checkpoint', 'resume', 'config'):
        path = Path(selection[field])
        record = row['file_records'][path.name]
        need(Path(record['path']) == path and sha(path) == record['sha256'], f'Selected file differs from training audit: {path}')
    need(row['file_records'][Path(selection['resume']).name]['sha256'] == selection['resume_sha256'], 'Resume SHA differs from selection')
    return {'full_audit': str(full_path), 'full_audit_sha256': selection['full_audit_sha256'],
            'training_audit': str(train_path), 'training_audit_sha256': selection['training_audit_sha256'],
            'run': run, 'global_step': selection['global_step'], 'independent_all514_FP32_finite_Adam4RNG': True}, {full_path, train_path}


def evaluation_guard(selection):
    directory = Path(selection['quality_dir']).resolve()
    need(directory.is_relative_to(ROOT) and directory.is_dir(), 'Quality dir outside root/missing')
    need((directory / 'render_complete').is_file(), 'Selected full renderer incomplete')
    manifest = read(directory / 'render_manifest.json')
    protocol = read(directory / 'render_protocol.json')
    need(manifest['protocol'] == protocol['protocol'] and manifest['protocol_id'] == protocol['protocol_id'] == identity(protocol['protocol']),
         'Render protocol identity differs')
    runner = protocol['protocol']['runner']
    need(runner['checkpoint_sha256'] == selection['checkpoint_sha256'] and Path(runner['weight_path']).resolve() == Path(selection['checkpoint']).resolve(),
         'Quality renderer checkpoint differs from selection')
    model_dir = Path(selection['model_dir']).resolve()
    need(model_dir.is_relative_to(ROOT / 'submissions') and model_dir.is_dir(), 'Selected model dir outside submissions')
    need(Path(protocol['protocol']['model_dir']).resolve() == model_dir, 'Selected model directory differs from full render')
    frozen_file(model_dir / 'runner.py', runner['runner_sha256'])
    for relative, digest in protocol['protocol']['model_dir_sha256'].items():
        if not transient(Path(relative)):
            p = under_root(model_dir / relative)
            need(p.is_relative_to(model_dir), 'Inference artifact escaped selected model directory')
            frozen_file(p, digest)
    renderer = Path(selection.get('renderer_source', ROOT / 'code/render_cuda_submission.py'))
    frozen_file(under_root(renderer), protocol['protocol']['source_sha256'])
    mcfg = read(model_dir / 'config.json')
    need(Path(mcfg['weight_path']).resolve() == Path(selection['checkpoint']).resolve(), 'Selected inference config checkpoint mismatch')
    for key, ckey in [('model_t', 'model_t'), ('coeff_t', 'coeff_t'), ('lora_rank', 'lora_rank')]:
        need(mcfg[ckey] == runner[key], f'Selected inference config/protocol differs: {key}')
    expected = [('val', n) for n in range(1, 4)] + [('test', n) for n in range(1, 101)]
    records = manifest['records']
    need([(r['split'], r['case']) for r in records] == expected and len({r['output'] for r in records}) == 103,
         'Selected render must contain ordered unique val3/test100')
    files = set()
    for task in protocol['protocol']['tasks']:
        frozen_file(under_root(task['input']), task['input_sha256'])
    for r in records:
        p = under_root(directory / r['output'])
        need(p.is_relative_to(directory) and p.suffix.lower() == '.png', 'Output must be a native PNG inside quality directory')
        frozen_file(p, r['sha256'])
        files.add(p)
    result = read(directory / 'scores/result.json')
    score_protocol = read(directory / 'scores/protocol.json')
    frozen_file(ROOT / 'code/score_cuda_stream.py', score_protocol['script_sha256'])
    for record in score_protocol['original_sources']:
        frozen_file(under_root(record['resolved_path']), record['sha256'])
    need(result.get('status') == 'complete' and result.get('completed') == result.get('expected') == 103
         and result.get('official_full_count') is True and result.get('S_proxy_scope') == 'full100_test_NR_plus3val_FR', 'Full scoring incomplete/scope differs')
    rows = [json.loads(line) for line in (directory / 'scores/measurements.jsonl').read_text().splitlines() if line.strip()]
    need([(r['split'], r['case']) for r in rows] == [('validation' if s == 'val' else s, n) for s, n in expected], 'Metric row ordering differs')
    means = {}
    scalar_count = 0
    for split, count, keys in [('validation', 3, ['lpips', 'dists', 'niqe', 'musiq', 'maniqa', 'clipiqa']),
                               ('test', 100, ['niqe', 'musiq', 'maniqa', 'clipiqa'])]:
        subset = [row for row in rows if row['split'] == split]
        need(len(subset) == count, 'Wrong metric split count')
        means[split] = {}
        for key in keys:
            values = [r['metrics'][key] for r in subset]
            need(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values), 'Nonfinite metric scalar')
            means[split][key] = sum(values) / count
            need(abs(means[split][key] - result[split][key]) <= 1e-12, 'Published component mean differs')
        scalar_count += count * len(keys)
    for row, record in zip(rows, records):
        need(row['sr_sha256'] == record['sha256'] and Path(row['sr_path']).resolve() == (directory / record['output']).resolve(), 'Scored PNG differs from manifest')
    need(scalar_count == 418, 'Full scalar count differs')
    test, val = means['test'], means['validation']
    nr = test['clipiqa'] + test['maniqa'] + test['musiq'] / 100 + max(0, (10 - test['niqe']) / 10)
    fr = 2 - val['lpips'] - val['dists']
    for key, value in [('NR', nr), ('FR', fr), ('S_proxy', nr + fr)]:
        need(abs(selection[key] - value) <= 1e-12, f'Selection {key} differs from actual full means')
    need(abs(result['test']['nr'] - nr) <= 1e-12 and abs(result['validation']['fr'] - fr) <= 1e-12
         and abs(result['S_proxy'] - nr - fr) <= 1e-12, 'Result NR/FR/S formula differs')
    return {'label': selection['label'], 'render_count': 103, 'metric_scalars': 418,
            'actual_png_manifest_scored_SHA_verified': True, 'NR': nr, 'FR': fr, 'S_proxy': nr + fr,
            'model_dir': str(model_dir), 'render_protocol_id': manifest['protocol_id']}, files, directory, model_dir


def transient(p):
    return '__pycache__' in p.parts or p.suffix in ('.pyc', '.tmp', '.lock')


def add_small(directory, files):
    if not directory.exists():
        return
    for p in directory.rglob('*'):
        if p.is_file() and not p.is_symlink() and not transient(p) and p.stat().st_size <= SMALL:
            if p.suffix.lower() not in ('.pt', '.pth', '.safetensors', '.ckpt', '.zip', '.png', '.jpg', '.jpeg', '.webp', '.gif'):
                files.add(p.resolve())


def source_guards(config):
    protocol = config['protocol']
    verified = {}
    for name, digest in protocol['source_sha256'].items():
        p = under_root(name)
        frozen_file(p, digest)
        verified[str(p)] = digest
    provenance = protocol['base_provenance']
    base = Path(provenance['cuda_base_path']).resolve()
    need(base == Path('/temp/zihaohe_stage/baselines/sd21b'), 'Shared base path changed')
    for category in ('initial_core_asset_hashes', 'all_config_tokenizer_scheduler_metadata_hashes'):
        for relative, record in provenance[category].items():
            frozen_file(base / relative, record['sha256'])
    return verified, {'path': str(base), 'provenance': provenance, 'weights_copied': False}


def fingerprint(path):
    p = under_root(path); before = token(p); digest = sha(p)
    need(token(p) == before, f'Source changed while hashing: {p}')
    return {'path': str(p.relative_to(ROOT)), 'bytes': before[0], 'sha256': digest}, before


def direct_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(data); stream.flush()
    need(path.read_bytes() == data, f'Metadata readback differs: {path}')


def stream_copy(source, target, record):
    before = token(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(); count = 0
    with source.open('rb') as reader, target.open('wb') as writer:
        for block in iter(lambda: reader.read(CHUNK), b''):
            writer.write(block); h.update(block); count += len(block)
        writer.flush()
    need(token(source) == before and count == record['bytes'] and h.hexdigest() == record['sha256'], 'Source changed during stream copy')
    need(target.stat().st_size == record['bytes'] and sha(target) == record['sha256'], f'SFS readback SHA/size differs: {target}')


def existing_match(path, record):
    return path.is_file() and not path.is_symlink() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']


def prepare(selection_path, reserve, plan_out=None, preview_cases=(7, 16, 41, 83)):
    FROZEN_TOKENS.clear()
    prior = previous_heads()
    selection_path = under_root(selection_path)
    need(selection_path == ROOT / 'results/nr_push4_selection.json', 'Selection must use final results/nr_push4_selection.json')
    selection = read(selection_path)
    need(selection['status'] == 'selected_after_independent_full_audit', 'Final audited selection status required')
    need(selection['global_step'] == 7250 + selection['local_step'] and selection['optimizer_step'] == 1250 + selection['local_step'], 'Selected stage step lineage differs')
    need(selection['local_step'] in (250, 500), 'Only immutable saved milestone selection allowed')
    config_path = under_root(selection['config']); run = config_path.parent
    need(run.parent == ROOT / 'adaptation/runs' and run.name.startswith('cuda_nr_stage5_'), 'Selection must name new stage5 run')
    weight = under_root(selection['checkpoint']); resume = under_root(selection['resume'])
    need(weight == run / f"step{selection['global_step']}.pth" and resume == run / f"resume_step{selection['global_step']}.pt", 'Selection must use matching raw and immutable milestone resume')
    frozen_file(weight, selection['checkpoint_sha256']); frozen_file(resume, selection['resume_sha256'])
    config = read(config_path)
    need(config['protocol_id'] == identity(config['protocol']) and config['protocol']['schema'] == 'cuda_raw_hypir_nr_stage_v5_best7250', 'New training protocol identity differs')
    need(selection['local_step'] <= config['target_stage_steps'], 'Selected step exceeds configured stage')
    marker_path = run / f"checkpoint_step{selection['global_step']}.json"
    marker = read(marker_path)
    need((marker['local_step'], marker['global_step'], marker['optimizer_step']) == (selection['local_step'], selection['global_step'], selection['optimizer_step']), 'Milestone marker steps differ')
    need(marker['raw_checkpoint_sha256'] == selection['checkpoint_sha256'] and marker['resume_checkpoint_sha256'] == selection['resume_sha256'], 'Milestone marker SHA differs')
    need((marker['resume_bytes'], marker['resume_mtime_ns'], marker['resume_inode']) == token(resume), 'Immutable selected resume stat changed')
    audit, files = audit_gates(selection)
    checkpoint = checkpoint_proof(weight, resume, config, selection)
    raw_recipe, recipe_files = exact_raw_recipe_proof(selection, resume)
    evaluation, pngs, quality_dir, model_dir = evaluation_guard(selection)
    sources, base_ref = source_guards(config)
    manifest = read(quality_dir / 'render_manifest.json')
    preview_records = [row for row in manifest['records'] if row['split'] == 'test' and row['case'] in preview_cases]
    need(len(preview_records) == len(preview_cases), 'Requested preview cases missing or duplicated')
    previews = {(quality_dir / row['output']).resolve() for row in preview_records}
    files |= recipe_files | previews | {selection_path, config_path, marker_path, resume, Path(__file__).resolve()}
    add_small(run, files)
    add_small(quality_dir, files)
    add_small(model_dir, files)
    for directory in (ROOT / 'code/push4', ROOT / 'audits/push4', ROOT / 'nr_push4_20261014/preparation'):
        add_small(directory, files)
    for directory, patterns in [(ROOT / 'reports', ['*PUSH4*', '*push4*']),
                                (ROOT / 'results', ['*push4*']), (ROOT / 'calibration', ['push4*'])]:
        for pattern in patterns:
            for p in directory.glob(pattern):
                if p.is_dir():add_small(p, files)
                elif p.is_file() and p.stat().st_size <= SMALL and not transient(p):files.add(p.resolve())
    # Only new source files needed by this selected run; old helpers already live in prior backups.
    files.add(ROOT / 'code/cuda_training/train_cuda_nr_stage_v5.py')
    # The actual new CLI configs are in the push4 subdirectory; require them.
    for name in ('config_stage5_A_continue500.json', 'config_stage5_B_m25_500.json',
                 'config_stage5_C_m25_r10_500.json', 'stage5_preparation_static_audit.json',
                 'STAGE_V5_README.md'):
        files.add(ROOT / 'code/cuda_training/push4' / name)
    for p in (ROOT / 'code/cuda_training/push4').glob('*'):
        if p.is_file() and p.stat().st_size <= SMALL and not transient(p):files.add(p.resolve())
    for p in (ROOT / 'logs').glob('push4*'):
        if p.is_file() and p.stat().st_size <= SMALL and not transient(p):files.add(p.resolve())
    if plan_out is not None:
        # A dry-run inventory must not recursively include its own generated plan.
        files.discard(Path(plan_out).resolve())
    records, signatures = [], {}
    for path in sorted(files):
        record, sig = fingerprint(path); records.append(record); signatures[str(path)] = sig
    signatures.update(FROZEN_TOKENS)
    archives = [r['path'] for r in records if Path(r['path']).suffix.lower() in ('.pt', '.pth', '.safetensors', '.ckpt')]
    need(archives == [str(resume.relative_to(ROOT))], 'Scope must retain only the one exact immutable resume, no duplicate raw or other tensor archives')
    actual_pngs = {ROOT / row['path'] for row in records if row['path'].endswith('.png')}
    need(actual_pngs == previews, 'Only explicitly selected preview PNGs may be persisted')
    actions = []
    for record in records:
        target = safe_target(record['path'])
        if not existing_match(target, record):
            previous_size = target.stat().st_size if target.is_file() else 0
            actions.append({'record': record, 'net_growth_bytes': max(0, record['bytes'] - previous_size)})
    net = sum(a['net_growth_bytes'] for a in actions)
    free = shutil.disk_usage(DEST if DEST.exists() else DEST.parent).free
    plan = {'schema': 'nr_push4_selected_backup_plan_v1', 'selection_path': str(selection_path),
            'selection_sha256': sha(selection_path), 'backup_script_sha256': sha(__file__),
            'source_root': str(ROOT), 'destination': str(DEST), 'prior_backup_heads': prior,
            'selected_run': run.name, 'selected_global_step': selection['global_step'],
            'selected_local_step': selection['local_step'], 'selected_optimizer_step': selection['optimizer_step'],
            'scope': 'selected_exact_resume_plus_exact_raw_recipe_and_optional_previews',
            'checkpoint_proof': checkpoint, 'raw_recipe_proof': raw_recipe,
            'independent_audit_proof': audit, 'full_evaluation_proof': evaluation,
            'source_full103_PNGs_actual_SHA_verified': True, 'full103_PNGs_persisted': False,
            'persisted_preview_cases': list(preview_cases), 'persisted_preview_PNG_count': len(previews),
            'raw_checkpoint_persisted': False, 'raw_exact_byte_recovery_available': True,
            'training_source_sha256_verified': sources, 'shared_SD2_reference': base_ref,
            'files': records, 'count': len(records), 'bytes': sum(r['bytes'] for r in records),
            'files_to_stream': len(actions), 'stream_bytes': sum(a['record']['bytes'] for a in actions),
            'net_growth_bytes': net, 'free_bytes_before': free, 'self_chosen_reserve_bytes': reserve,
            'reserve_is_user_requirement': False, 'metadata_allowance_bytes': ALLOWANCE,
            'projected_free_after_bytes': free - net, 'space_sufficient': free >= net + reserve + ALLOWANCE,
            'exclusions': ['duplicate selected raw checkpoint (exact original bytes reconstructable from retained resume plus recipe)',
                           'all other candidate/milestone weights and resumes', 'all native full/pilot PNGs except explicitly selected previews; all full outputs remain on /cache',
                           'env', 'shared SD2 base weights', 'old-prefix payload already backed up', 'unscoped old reports/code',
                           'mutable resume_latest.pt', 'transient lock/tmp/pyc'],
            'copy_method': 'direct final-path bytes, source SHA while streaming, entire SFS SHA readback; resume only actual SHA+size match'}
    return plan, actions, signatures


def execute(plan, actions, signatures, reserve):
    need(plan['space_sufficient'], 'Insufficient space for selected-only backup plus self-chosen reserve')
    if (DEST / 'DONE').exists():
        old_done = read(DEST / 'DONE'); old_manifest = read(DEST / 'manifest.json')
        need(old_done['status'] == 'complete' and sha(DEST / 'manifest.json') == old_done['manifest_sha256'], 'Existing new-prefix commit damaged')
        need(old_manifest['selection_sha256'] == plan['selection_sha256'], 'New-prefix commit already belongs to another final selection')
        need(old_manifest['files'] == plan['files'], 'Existing committed file inventory changed; keep committed prefix immutable')
        need(not actions, 'Existing committed backup has missing/corrupted data')
        print(json.dumps({'already_committed_and_all_bytes_reverified': True, 'DONE': old_done}), flush=True)
        return
    remaining = sum(a['net_growth_bytes'] for a in actions)
    for action in actions:
        free = shutil.disk_usage(DEST if DEST.exists() else DEST.parent).free
        need(free >= remaining + reserve + ALLOWANCE, 'SFS available space changed; no deletion performed')
        record = action['record']; stream_copy(ROOT / record['path'], safe_target(record['path']), record)
        remaining -= action['net_growth_bytes']
        print(json.dumps({'copied_and_full_SHA_readback_verified': record['path'], 'bytes': record['bytes'],
                          'remaining_net_growth_bytes': remaining}), flush=True)
    for path, before in signatures.items():
        need(token(path) == before, f'Source changed before commit: {path}')
    need(previous_heads() == plan['prior_backup_heads'], 'Previous backup heads changed before commit')
    need(shutil.disk_usage(DEST).free >= reserve + ALLOWANCE, 'Self-chosen reserve unavailable before commit')
    manifest = dict(plan, schema='nr_push4_selected_backup_manifest_v1', status='complete',
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    all_selected_destination_bytes_full_SHA_readback_verified=True)
    data = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    direct_write(DEST / 'manifest.json', data)
    digest = hashlib.sha256(data).hexdigest()
    done = {'schema': 'nr_push4_selected_backup_commit_v1', 'status': 'complete',
            'destination': str(DEST), 'scope': plan['scope'], 'selection_sha256': plan['selection_sha256'],
            'manifest_sha256': digest, 'files': plan['count'], 'bytes': plan['bytes'],
            'selected_global_step': plan['selected_global_step'], 'selected_local_step': plan['selected_local_step'],
            'full103_PNGs_persisted': False, 'persisted_preview_PNG_count': plan['persisted_preview_PNG_count'],
            'raw_checkpoint_persisted': False, 'raw_exact_byte_recovery_available': True,
            'self_chosen_reserve_bytes': reserve, 'reserve_is_user_requirement': False}
    direct_write(DEST / 'DONE', (json.dumps(done, indent=2) + '\n').encode())
    print(json.dumps({'commit_complete': done, 'free_bytes_after': shutil.disk_usage(DEST).free}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection-json', type=Path, default=ROOT / 'results/nr_push4_selection.json')
    ap.add_argument('--reserve-gib', type=float, default=.25, help='Self-chosen free-space margin, not a user requirement')
    ap.add_argument('--plan-out', type=Path, help='Optionally write reviewed plan JSON on /cache; excluded from its own payload')
    ap.add_argument('--preview-cases', default='7,16,41,83', help='Four optional native test previews, or none; full103 never persisted')
    ap.add_argument('--execute', action='store_true', help='Actually stream data and commit; otherwise dry-run')
    args = ap.parse_args()
    need(math.isfinite(args.reserve_gib) and 0 <= args.reserve_gib <= 20, 'Reserve GiB must be finite0..20')
    reserve = int(args.reserve_gib * 1024**3)
    preview_cases = [] if args.preview_cases.lower() == 'none' else [int(x) for x in args.preview_cases.split(',')]
    need(len(preview_cases) in (0, 4) and len(set(preview_cases)) == len(preview_cases)
         and all(1 <= x <= 100 for x in preview_cases), 'Choose exactly four distinct test cases1..100, or none')
    lock_path = ROOT / 'device_locks/push4_selected_backup.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, actions, signatures = prepare(args.selection_json, reserve, args.plan_out, preview_cases)
        print(json.dumps({'backup_plan': plan, 'dry_run': not args.execute}, ensure_ascii=False), flush=True)
        if args.plan_out:
            p = args.plan_out.resolve()
            need(p.is_relative_to(ROOT) and p.parent != DEST, 'Plan output must be in experiment /cache')
            need(str(p) not in signatures, 'Plan output cannot overwrite a selected source')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
        if args.execute:
            execute(plan, actions, signatures, reserve)
        else:
            print(json.dumps({'dry_run_complete': True, 'copies_started': False, 'space_sufficient': plan['space_sufficient']}), flush=True)


if __name__ == '__main__':
    main()
