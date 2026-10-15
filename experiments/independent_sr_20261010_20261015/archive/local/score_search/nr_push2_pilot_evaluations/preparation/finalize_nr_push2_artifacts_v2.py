"""Append stage3 checkpoints or final evaluations to the verified stage2 SFS backup.

V2 exposes an explicit self-chosen reserve in GiB without altering the frozen v1.
Default is a read-only dry run. --execute explicitly copies. Standard library only;
no torch/NumPy imports, accelerators, deletion, permission changes or SFS renames.
Previous commit metadata and superseded selected files are preserved by checksum.
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
import os
from pathlib import Path
import pickle
import shutil
import struct
import zipfile

ROOT = Path('/cache/independent_sr_cuda_20261013')
DEFAULT_PLAN = ROOT / 'nr_push2_pilot_evaluations/preparation/checkpoint_backup_plan.json'
CHUNK = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


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
    """Whitelist inert tensor/NumPy references; never imports those packages."""

    def find_class(self, module, name):
        pair = (module, name)
        if pair == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if pair == ('torch._utils', '_rebuild_tensor_v2'):
            return TensorReference
        if module == 'torch' and name in ('FloatStorage', 'ByteStorage'):
            return name
        if pair == ('_codecs', 'encode'):
            return codecs.encode
        if pair in (('numpy', 'dtype'), ('numpy', 'ndarray'),
                    ('numpy._core.multiarray', '_reconstruct')):
            return InertNumpyObject
        raise ValueError(f'Unsupported checkpoint pickle global: {module}.{name}')

    def persistent_load(self, identity):
        require(isinstance(identity, tuple) and len(identity) == 5
                and identity[0] == 'storage', 'Unsupported persistent storage reference')
        _, kind, key, location, count = identity
        require(kind in ('FloatStorage', 'ByteStorage'), 'Unsupported storage dtype')
        require(isinstance(key, str) and key.isdigit(), 'Invalid storage key')
        require(location == 'cpu' or location.startswith('cuda:'), 'Unknown storage location')
        return StorageReference(kind, key, location, count)


def checkpoint_proof(path, expected_protocol):
    with zipfile.ZipFile(path) as archive:
        candidates = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        require(len(candidates) == 1, f'{path}: ambiguous data.pkl')
        name = candidates[0]
        prefix = name[:-len('data.pkl')]
        require(archive.getinfo(name).file_size <= 2 * 1024 * 1024,
                f'{path}: unexpectedly large checkpoint metadata')
        root = MetadataOnlyUnpickler(io.BytesIO(archive.read(name))).load()
        expected = {'local_step': 250, 'global_step': 6500, 'optimizer_step': 500}
        require(all(root.get(k) == v for k, v in expected.items()),
                f'{path}: checkpoint step counters differ from {expected}')
        require(root.get('protocol_id') == expected_protocol, f'{path}: protocol mismatch')
        lora = root['lora']
        states = root['optimizer']['state']
        require(len(lora) == len(states) == 514, f'{path}: expected 514 LoRA/Adam states')
        require(all(isinstance(t, TensorReference) and t.storage.kind == 'FloatStorage'
                    for t in lora.values()), f'{path}: LoRA is not FP32')
        steps = []
        byteorder_name = prefix + 'byteorder'
        endian = '<'
        if byteorder_name in archive.namelist():
            byteorder = archive.read(byteorder_name).decode().strip()
            require(byteorder in ('little', 'big'), 'Unsupported checkpoint byte order')
            endian = '<' if byteorder == 'little' else '>'
        for state in states.values():
            tensor = state['step']
            require(isinstance(tensor, TensorReference) and tensor.size == ()
                    and tensor.storage.kind == 'FloatStorage',
                    f'{path}: Adam step is not a scalar FP32 tensor')
            with archive.open(prefix + 'data/' + tensor.storage.key) as stream:
                stream.read(tensor.offset * 4)
                value = struct.unpack(endian + 'f', stream.read(4))[0]
            require(value == 500.0, f'{path}: Adam tensor step {value} != 500')
            steps.append(value)
        require(all(k in root for k in ('python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng')),
                f'{path}: missing restart RNG state')
    return {**expected, 'lora_fp32_tensors': len(lora), 'adam_states': len(steps),
            'all_adam_tensor_steps': 500.0, 'all_four_rng_states_present': True,
            'method': 'stdlib ZIP + whitelisted inert pickle references + actual FP32 scalar storage reads; no torch import'}


def run_proof(run_name):
    directory = ROOT / 'adaptation/runs' / run_name
    status = json.loads((directory / 'status.json').read_text())
    config = json.loads((directory / 'config.json').read_text())
    expected_status = {'stage': 'complete', 'completed_local_steps': 250,
                       'global_step': 6500, 'optimizer_step': 500}
    require(all(status.get(k) == v for k, v in expected_status.items()),
            f'{run_name}: training is not complete at local250/global6500/Adam500')
    require((directory / 'training_complete_step6500').is_file(),
            f'{run_name}: completion marker missing')
    require(config.get('target_stage_steps') == 250
            and config.get('target_global_step') == 6500
            and config.get('target_optimizer_step') == 500,
            f'{run_name}: target counters differ')
    require(config['protocol']['optimizer'].startswith('AdamW'),
            f'{run_name}: optimizer is not recorded as AdamW')
    rows = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()
            if line.strip()]
    require(len(rows) == 250, f'{run_name}: expected 250 training log rows')
    for index, row in enumerate(rows, 1):
        require((row.get('local_step'), row.get('global_step'), row.get('optimizer_step'))
                == (index, 6250 + index, 250 + index),
                f'{run_name}: discontinuous training/optimizer counters at row {index}')
    gradient = json.loads((directory / 'gradient_preflight.json').read_text())
    require(gradient.get('status') == 'passed', f'{run_name}: gradient preflight did not pass')
    for path, digest in config['protocol']['source_sha256'].items():
        require(sha(path) == digest, f'{run_name}: training source changed: {path}')
    migration = json.loads((directory / 'parent_migration.json').read_text())
    require(migration.get('FP32_LoRA_exact_equal_before_update') is True,
            f'{run_name}: parent migration equality proof absent')
    require(migration.get('restored_global_step') == 6250
            and migration.get('restored_optimizer_step') == 250,
            f'{run_name}: parent counters differ')
    proof = checkpoint_proof(directory / 'resume_latest.pt', config['protocol_id'])
    return {'run': run_name, 'status': status, 'protocol_id': config['protocol_id'],
            'training_log_rows': len(rows), 'gradient_preflight': 'passed',
            'source_sha256_verified': config['protocol']['source_sha256'],
            'parent_lineage': migration['parent_lineage'], 'resume_checkpoint': proof}


def safe_target(destination, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,
            f'Unsafe destination relative path: {relative}')
    path = destination / relative
    require(not destination.is_symlink(), 'Destination must not be a symlink')
    for ancestor in [path, *path.parents]:
        if ancestor == destination.parent:
            break
        require(not ancestor.is_symlink(), f'Symlink destination path: {ancestor}')
    return path


def direct_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(data)
        stream.flush()
    require(path.read_bytes() == data, f'Written metadata readback failed: {path}')


def verified_stream_copy(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    signature_before = (source.stat().st_size, source.stat().st_mtime_ns)
    digest = hashlib.sha256()
    written = 0
    # Direct stream to final SFS path. No copy2, chmod, timestamp, rename or unlink.
    with source.open('rb') as reader, target.open('wb') as writer:
        for block in iter(lambda: reader.read(CHUNK), b''):
            writer.write(block)
            digest.update(block)
            written += len(block)
        writer.flush()
    require(signature_before == (source.stat().st_size, source.stat().st_mtime_ns),
            f'Source changed during copy: {source}')
    require(written == expected['bytes'] and digest.hexdigest() == expected['sha256'],
            f'Source stream SHA/size changed: {source}')
    require(target.stat().st_size == written and sha(target) == expected['sha256'],
            f'Destination readback SHA/size mismatch: {target}')


DESTINATION = Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014')
GENESIS_MANIFEST_SHA = 'b54f98027dad5cc30c2914f2b3791a45e4a66ab509d732f0ecbae043ad8fd8ed'
FROZEN_BACKUP_SCRIPT_SHA = '64224c1a274823155b1b75643e042ae3629e7f423afcd47d4826b2e8e14c77ad'
FROZEN_BACKUP_PLAN_SHA = 'f92497551985a9c2ca087954577fe6a875907676699ad67e060ce2ad69bf4a74'
RESERVE = 10 * 1024 ** 3
METADATA_ALLOWANCE = 16 * 1024 ** 2
SMALL_LIMIT = 16 * 1024 ** 2
TRUSTED_LARGE_LIMIT = 64 * 1024 ** 2
STAGE3_RUNS = ['cuda_nr_stage3_s6250_anchor6250_m1_250',
               'cuda_nr_stage3_s6250_anchor6250_m15_250']
PILOT_CASES = [1, 6, 7, 11, 16, 18, 21, 23, 25, 28, 31, 36, 41, 46, 51, 61, 71, 81, 91, 100]
REQUIRED_PILOTS = {
    'nr_s6250_ct160_compiled_pilot', 'nr_s6250_ct175_freshcontrol_pilot',
    'nr_s6250_ct185_compiled_pilot', 'nr_stage2_cont_s6350_compiled_pilot',
    'nr_stage2_cont_s6500_compiled_pilot', 'nr_stage2_roundste_s6500_compiled_pilot',
    'nr_stage3_anchor6250_m1_s6500_compiled_pilot',
    'nr_stage3_anchor6250_m15_s6500_compiled_pilot',
}


def fingerprint(path):
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    digest = sha(path)
    require(before == (path.stat().st_size, path.stat().st_mtime_ns),
            f'Source changed during hashing: {path}')
    return {'path': str(path.relative_to(ROOT)), 'bytes': before[0], 'sha256': digest}, before


def read_committed_head():
    require(DESTINATION.is_dir(), 'Verified stage2 destination does not exist')
    manifest_bytes = (DESTINATION / 'manifest.json').read_bytes()
    done_bytes = (DESTINATION / 'DONE').read_bytes()
    manifest, done = json.loads(manifest_bytes), json.loads(done_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    require(done.get('status') == 'complete' and done.get('manifest_sha256') == digest,
            'Root manifest/DONE checksum differs; preserved .commits metadata is available for recovery')
    require(manifest.get('source_root') == str(ROOT)
            and manifest.get('destination') == str(DESTINATION), 'Committed backup roots differ')
    require(len(manifest['files']) == len({r['path'] for r in manifest['files']}),
            'Duplicate records in committed manifest')
    chain, seen = [], set()
    current, current_digest = manifest, digest
    while True:
        require(current_digest not in seen, 'Commit checksum chain contains a cycle')
        seen.add(current_digest)
        chain.append(current_digest)
        if current_digest == GENESIS_MANIFEST_SHA:
            genesis = current
            break
        previous = current.get('previous_commit')
        require(isinstance(previous, dict), 'Commit chain does not reach the verified stage2 genesis')
        parent_manifest_path = safe_target(DESTINATION, previous['manifest_path'])
        parent_done_path = safe_target(DESTINATION, previous['DONE_path'])
        parent_manifest_bytes = parent_manifest_path.read_bytes()
        parent_done_bytes = parent_done_path.read_bytes()
        require(hashlib.sha256(parent_manifest_bytes).hexdigest() == previous['manifest_sha256'],
                'Archived parent manifest checksum differs')
        require(hashlib.sha256(parent_done_bytes).hexdigest() == previous['DONE_sha256'],
                'Archived parent DONE checksum differs')
        require(json.loads(parent_done_bytes)['manifest_sha256'] == previous['manifest_sha256'],
                'Archived parent DONE does not name its manifest checksum')
        current, current_digest = json.loads(parent_manifest_bytes), previous['manifest_sha256']
    require(genesis['backup_script_sha256'] == FROZEN_BACKUP_SCRIPT_SHA
            and genesis['backup_plan_sha256'] == FROZEN_BACKUP_PLAN_SHA,
            'Genesis frozen backup source/plan identity differs')
    require(sha(ROOT / 'code/sync_nr_push2_artifacts.py') == FROZEN_BACKUP_SCRIPT_SHA,
            'Frozen checkpoint backup source changed')
    require(sha(DEFAULT_PLAN) == FROZEN_BACKUP_PLAN_SHA, 'Frozen checkpoint backup plan changed')
    return manifest, manifest_bytes, done_bytes, digest, chain, genesis


def is_transient(path):
    return '__pycache__' in path.parts or path.suffix in ('.pyc', '.tmp', '.lock')


def collect_small(directory, paths, excluded):
    if not directory.exists():
        return
    for path in directory.rglob('*'):
        if not path.is_file() or path.is_symlink() or is_transient(path):
            continue
        if path.stat().st_size <= SMALL_LIMIT:
            paths.add(path)
        else:
            excluded.append(str(path.relative_to(ROOT)))


def collect_sources(checkpoint_only):
    paths, excluded = set(), []
    for name in STAGE3_RUNS:
        directory = ROOT / 'adaptation/runs' / name
        paths.update(directory / n for n in ('step6500.pth', 'resume_latest.pt'))
        if directory.exists():
            for path in directory.rglob('*'):
                if not path.is_file() or path.is_symlink() or is_transient(path):
                    continue
                if path.suffix in ('.pt', '.pth') and path.name not in ('step6500.pth', 'resume_latest.pt'):
                    excluded.append(str(path.relative_to(ROOT)))
                elif path.name not in ('step6500.pth', 'resume_latest.pt'):
                    require(path.stat().st_size <= SMALL_LIMIT,
                            f'Unexpected large stage3 small-record file: {path}')
                    paths.add(path)
        config_path = directory / 'config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            paths.update(Path(p) for p in config['protocol']['source_sha256'])
    collect_small(ROOT / 'code/cuda_training', paths, excluded)
    paths.add(Path(__file__).resolve())
    readme = ROOT / 'code/FINALIZE_NR_PUSH2_README.txt'
    if readme.exists():
        paths.add(readme)
    for name in ('push2_train_stage3_m1.log', 'push2_train_stage3_m15.log',
                 'push2_train_anchor6250_m1.log', 'push2_train_anchor6250_m15.log'):
        path = ROOT / 'logs' / name
        if path.exists():
            paths.add(path)
    for directory in (ROOT / 'submissions').glob('hypir_cuda_nr_stage3_*_compiled'):
        collect_small(directory, paths, excluded)
    if not checkpoint_only:
        for name in ('nr_push2_pilot_evaluations', 'nr_push2_full_evaluations'):
            for path in (ROOT / name).rglob('*'):
                if path.is_file() and not path.is_symlink() and not is_transient(path):
                    paths.add(path)
        for name in ('code', 'audits', 'reports', 'results', 'submissions'):
            collect_small(ROOT / name, paths, excluded)
        for path in (ROOT / 'logs').glob('push2*'):
            if path.is_file() and path.stat().st_size <= SMALL_LIMIT and not is_transient(path):
                paths.add(path)
    return paths, excluded


def evaluation_proof(directory, cases):
    require((directory / 'render_complete').is_file(), f'{directory.name}: renderer incomplete')
    manifest = json.loads((directory / 'render_manifest.json').read_text())
    expected_render = [('val', n) for n in range(1, 4)] + [('test', n) for n in cases]
    require([(r['split'], r['case']) for r in manifest['records']] == expected_render,
            f'{directory.name}: render sequence/count differs')
    require(len({r['output'] for r in manifest['records']}) == len(expected_render),
            f'{directory.name}: duplicate render output paths')
    for record in manifest['records']:
        path = directory / record['output']
        require(path.is_file() and not path.is_symlink(), f'Missing output PNG: {path}')
        require(sha(path) == record['sha256'], f'Rendered PNG differs from its manifest: {path}')
    result = json.loads((directory / 'scores/result.json').read_text())
    require(result.get('status') == 'complete'
            and result.get('completed') == result.get('expected') == len(expected_render),
            f'{directory.name}: scoring incomplete')
    require(result['test']['n'] == len(cases) and result['validation']['n'] == 3,
            f'{directory.name}: score split counts differ')
    full = len(cases) == 100
    require(result.get('official_full_count') is full, f'{directory.name}: full/subset flag differs')
    if full:
        require(result.get('S_proxy_scope') == 'full100_test_NR_plus3val_FR',
                f'{directory.name}: full score scope differs')
    rows = [json.loads(line) for line in (directory / 'scores/measurements.jsonl').read_text().splitlines()
            if line.strip()]
    expected_score = [('validation' if split == 'val' else split, case)
                      for split, case in expected_render]
    require([(row['split'], row['case']) for row in rows] == expected_score,
            f'{directory.name}: metric sequence/count differs')
    for row, rendered in zip(rows, manifest['records']):
        require(row['sr_sha256'] == rendered['sha256'],
                f'{directory.name}: scored PNG SHA does not match render manifest')
        require(Path(row['sr_path']).resolve() == (directory / rendered['output']).resolve(),
                f'{directory.name}: metric row references a different PNG path')
    return {'label': directory.name, 'render_count': len(expected_render),
            'test_count': len(cases), 'score_name': 'S_full' if full else 'S_subset',
            'complete': True, 'render_protocol_id': manifest['protocol_id']}


def evaluation_guards():
    pilot_root, full_root = ROOT / 'nr_push2_pilot_evaluations', ROOT / 'nr_push2_full_evaluations'
    pilots = [p for p in pilot_root.iterdir() if p.is_dir() and p.name != 'preparation']
    require(REQUIRED_PILOTS <= {p.name for p in pilots},
            f'Missing required pilots: {sorted(REQUIRED_PILOTS - {p.name for p in pilots})}')
    fulls = [p for p in full_root.iterdir() if p.is_dir()]
    require(fulls, 'No formal full evaluation directories exist')
    return ([evaluation_proof(p, PILOT_CASES) for p in sorted(pilots)]
            + [evaluation_proof(p, list(range(1, 101))) for p in sorted(fulls)])


def existing_match(path, record, trust_large=False):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != record['bytes']:
        return False
    if trust_large and record['bytes'] >= TRUSTED_LARGE_LIMIT:
        return True
    return sha(path) == record['sha256']


def prepare_actions(old, sources, signatures, old_manifest_bytes, old_done_bytes, old_sha):
    merged = {row['path']: dict(row) for row in old['files']}
    require(len(merged) == len(old['files']), 'Duplicate previous records')
    actions, selected, inherited_large = [], {row['path'] for row in sources}, 0
    commit_dir = f'.commits/{old_sha}'
    snapshots = [(f'{commit_dir}/manifest.json', old_manifest_bytes),
                 (f'{commit_dir}/DONE', old_done_bytes)]
    for relative, data in snapshots:
        record = {'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        target = safe_target(DESTINATION, relative)
        if not existing_match(target, record):
            actions.append({'kind': 'bytes', 'target': target, 'data': data, 'record': record})
        merged[relative] = record
    # Unselected previous data remains part of every new manifest. Already verified
    # large files retain their records without another complete multi-GB read.
    for relative, row in list(merged.items()):
        if relative in selected or relative.startswith(commit_dir + '/'):
            continue
        require(existing_match(safe_target(DESTINATION, relative), row, trust_large=True),
                f'Previously committed file missing or changed: {relative}')
        inherited_large += row['bytes'] >= TRUSTED_LARGE_LIMIT
    for record in sources:
        relative = record['path']
        target = safe_target(DESTINATION, relative)
        prior = merged.get(relative)
        same = prior and prior['bytes'] == record['bytes'] and prior['sha256'] == record['sha256']
        if same and existing_match(target, prior, trust_large=True):
            inherited_large += prior['bytes'] >= TRUSTED_LARGE_LIMIT
        else:
            if prior and not same:
                archived = dict(prior, path=f'{commit_dir}/superseded/{relative}')
                archive_target = safe_target(DESTINATION, archived['path'])
                if not existing_match(archive_target, archived):
                    require(existing_match(target, prior),
                            f'Changed prior data has no verified preserved version: {relative}')
                    actions.append({'kind': 'copy', 'source': target, 'target': archive_target,
                                    'record': archived})
                merged[archived['path']] = archived
            if not existing_match(target, record):
                actions.append({'kind': 'copy', 'source': ROOT / relative,
                                'target': target, 'record': record})
        merged[relative] = dict(record)
    return merged, actions, inherited_large, commit_dir


def finalize(checkpoint_only, execute):
    old, old_manifest_bytes, old_done_bytes, old_sha, chain, genesis = read_committed_head()
    paths, excluded = collect_sources(checkpoint_only)
    missing = sorted(str(p) for p in paths if not p.is_file())
    present_bytes = sum(p.stat().st_size for p in paths if p.is_file())
    free = shutil.disk_usage(DESTINATION).free
    statuses = {}
    for name in STAGE3_RUNS:
        path = ROOT / 'adaptation/runs' / name / 'status.json'
        statuses[name] = json.loads(path.read_text()) if path.exists() else 'missing'
    print(json.dumps({'preflight_inventory': True, 'mode': 'checkpoint_only' if checkpoint_only else 'final',
                      'execution': execute, 'source_files_present': len(paths) - len(missing),
                      'selected_source_bytes_before_deduplication': present_bytes,
                      'missing_required_files': missing, 'previous_committed_files': len(old['files']),
                      'previous_committed_bytes': old['bytes'], 'free_bytes': free,
                      'self_chosen_reserve_bytes': RESERVE, 'reserve_is_user_requirement': False,
                      'reserve_note': f'{RESERVE // 1024**3}GiB for append/finalize; initial20GiB was only the earlier checkpoint phase reserve',
                      'stage3_status': statuses}), flush=True)
    # Both checkpoint-only and final operations reject unfinished stage3 runs.
    proofs = [run_proof(name) for name in STAGE3_RUNS]
    require(not missing, f'Missing required source files: {missing}')
    evaluation_proofs = [] if checkpoint_only else evaluation_guards()
    source_records, signatures = [], {}
    for path in sorted(paths):
        require(path.is_file() and not path.is_symlink(), f'Unsupported source path: {path}')
        record, signature = fingerprint(path)
        source_records.append(record)
        signatures[str(path)] = signature
    merged, actions, inherited_large, commit_dir = prepare_actions(
        old, source_records, signatures, old_manifest_bytes, old_done_bytes, old_sha)
    needed = sum(action['record']['bytes'] for action in actions)
    free = shutil.disk_usage(DESTINATION).free
    sufficient = free >= needed + RESERVE + METADATA_ALLOWANCE
    print(json.dumps({'copy_plan': True, 'actions': len(actions), 'bytes_to_write': needed,
                      'merged_record_count': len(merged), 'merged_record_bytes': sum(r['bytes'] for r in merged.values()),
                      'inherited_large_records_without_full_rehash': inherited_large,
                      'free_bytes': free, 'self_chosen_reserve_bytes': RESERVE,
                      'metadata_allowance_bytes': METADATA_ALLOWANCE,
                      'projected_free_after_data_bytes': free - needed, 'space_sufficient': sufficient,
                      'checkpoint_guards': [{'run': p['run'], **p['resume_checkpoint']} for p in proofs],
                      'evaluation_guards': evaluation_proofs}), flush=True)
    require(sufficient, f'Insufficient SFS space: free={free}, needed={needed}, '
            f'self-chosen reserve_bytes={RESERVE}, metadata_allowance={METADATA_ALLOWANCE}; no data deleted')
    if not execute:
        print(json.dumps({'dry_run_complete': True, 'copies_started': False}), flush=True)
        return
    # The checksummed old metadata is preserved before any selected old data changes.
    remaining, verified_targets = needed, set()
    for action in actions:
        available = shutil.disk_usage(DESTINATION).free
        require(available >= remaining + RESERVE + METADATA_ALLOWANCE,
                f'SFS free space changed: free={available}, remaining={remaining}, '
                f'self-chosen reserve={RESERVE}; no data deleted')
        if action['kind'] == 'bytes':
            direct_write(action['target'], action['data'])
            require(sha(action['target']) == action['record']['sha256'], 'Commit metadata archive SHA mismatch')
        else:
            verified_stream_copy(action['source'], action['target'], action['record'])
        remaining -= action['record']['bytes']
        verified_targets.add(action['record']['path'])
        print(json.dumps({'copied_and_sha_verified': action['record']['path'],
                          'bytes': action['record']['bytes'], 'remaining_bytes': remaining}), flush=True)
    for path, signature in signatures.items():
        path = Path(path)
        require((path.stat().st_size, path.stat().st_mtime_ns) == signature,
                f'Source changed before commit: {path}')
    require((DESTINATION / 'manifest.json').read_bytes() == old_manifest_bytes
            and (DESTINATION / 'DONE').read_bytes() == old_done_bytes,
            'Committed head changed concurrently; no new DONE written')
    final_proofs = [run_proof(name) for name in STAGE3_RUNS]
    if not checkpoint_only:
        evaluation_guards()
    require(shutil.disk_usage(DESTINATION).free >= RESERVE + METADATA_ALLOWANCE,
            'Self-chosen reserve unavailable before metadata commit; no new DONE written')
    previous = {'manifest_sha256': old_sha,
                'manifest_path': f'{commit_dir}/manifest.json',
                'DONE_path': f'{commit_dir}/DONE',
                'DONE_sha256': hashlib.sha256(old_done_bytes).hexdigest()}
    all_proofs = {p['run']: p for p in old.get('run_proofs', [])}
    all_proofs.update({p['run']: p for p in final_proofs})
    manifest = {'schema': 'nr_push2_append_backup_v1', 'status': 'complete',
                'source_root': str(ROOT), 'destination': str(DESTINATION),
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scope': 'stage3_checkpoint_append' if checkpoint_only else 'final_all_pilots_and_formal_full_artifacts',
                'final_quality_and_render_artifacts_pending_separate_finalize': checkpoint_only,
                'previous_commit': previous, 'genesis_manifest_sha256': GENESIS_MANIFEST_SHA,
                'parent_reference': genesis['parent_reference'], 'run_proofs': list(all_proofs.values()),
                'evaluation_proofs': evaluation_proofs,
                'files': [merged[p] for p in sorted(merged)], 'count': len(merged),
                'bytes': sum(row['bytes'] for row in merged.values()),
                'self_chosen_reserve_bytes': RESERVE, 'reserve_is_user_requirement': False,
                'reserve_note': f'{RESERVE // 1024**3}GiB chosen for this append/finalize phase; earlier20GiB was phase-specific, not a user limit',
                'previous_commit_count': len(chain),
                'previous_large_records_retained_without_full_rehash': inherited_large,
                'new_or_changed_files_sha_readback_verified': len(verified_targets),
                'frozen_checkpoint_backup_script_sha256': FROZEN_BACKUP_SCRIPT_SHA,
                'frozen_checkpoint_backup_plan_sha256': FROZEN_BACKUP_PLAN_SHA,
                'finalizer_source_sha256': sha(Path(__file__)),
                'exclusions': excluded + ['parent prefix data already backed up', 'SD2 base weights', 'env',
                                           'intermediate stage3 step weights', 'transient lock/tmp/pyc files'],
                'copy_method': 'direct final-path byte streams + SHA256 readback; no copy2, chmod, rename or deletion'}
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
    direct_write(DESTINATION / 'manifest.json', manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    done = {'status': 'complete', 'scope': manifest['scope'], 'files': len(merged),
            'bytes': manifest['bytes'], 'manifest_sha256': manifest_sha,
            'previous_manifest_sha256': old_sha,
            'previous_DONE_sha256': previous['DONE_sha256'],
            'genesis_manifest_sha256': GENESIS_MANIFEST_SHA,
            'destination': str(DESTINATION), 'self_chosen_reserve_bytes': RESERVE,
            'reserve_is_user_requirement': False, 'finalize_pending': checkpoint_only}
    direct_write(DESTINATION / 'DONE', (json.dumps(done, indent=2) + '\n').encode())
    print(json.dumps({'commit_complete': done, 'free_bytes_after': shutil.disk_usage(DESTINATION).free}), flush=True)


def main():
    global RESERVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-only', action='store_true', help='Append completed stage3 runs/source without requiring evaluation completion')
    parser.add_argument('--execute', action='store_true', help='Actually copy/commit; omitted means read-only dry run')
    parser.add_argument('--reserve-gib', type=int, default=10, help='Explicit self-chosen free-space reserve in GiB (default10, not a user requirement)')
    args = parser.parse_args()
    if not 1 <= args.reserve_gib <= 100:
        parser.error('--reserve-gib must be between1 and100')
    RESERVE = args.reserve_gib * 1024 ** 3
    if args.execute:
        path = ROOT / 'device_locks/nr_push2_finalize.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finalize(args.checkpoint_only, True)
    else:
        finalize(args.checkpoint_only, False)


if __name__ == '__main__':
    main()
