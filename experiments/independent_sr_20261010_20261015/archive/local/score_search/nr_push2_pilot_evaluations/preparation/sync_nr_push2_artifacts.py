"""Checkpoint-only SFS backup. Default: verify/plan without copying; --execute copies.

Standard library only. Does not import torch/numpy, initialize accelerators, delete
files, change permissions, preserve source metadata, or rename SFS files.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--execute', action='store_true', help='Actually copy verified files to SFS')
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    require(plan['source_root'] == str(ROOT), 'Unexpected source root')
    require(sha(Path(__file__)) == plan['script_sha256'], 'Backup script SHA differs from pinned plan')
    destination = Path(plan['destination'])
    require(destination == Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014'),
            'Unexpected SFS destination')
    parent = Path(plan['parent_reference']['prefix'])
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    require(sha(parent / 'manifest.json') == plan['parent_reference']['manifest_sha256'],
            'Previously backed up parent manifest changed')
    require((parent / 'DONE.json').is_file(), 'Previously backed up parent DONE marker missing')
    proofs = [run_proof(name) for name in plan['runs']]
    parent_files = {row['path']: row for row in parent_manifest['files']}
    for proof in proofs:
        lineage = proof['parent_lineage']
        parent_relative = str(Path(lineage['parent_resume']).relative_to(ROOT))
        require(parent_relative in parent_files
                and parent_files[parent_relative]['sha256'] == lineage['parent_resume_sha256'],
                'Existing parent backup does not record the referenced parent resume SHA')
    entries = list(plan['files']) + [{'path': 'checkpoint_backup_plan.json',
                                    'source': str(plan_path), 'bytes': len(plan_bytes),
                                    'sha256': hashlib.sha256(plan_bytes).hexdigest()}]
    require(len({entry['path'] for entry in entries}) == len(entries), 'Duplicate destination paths')
    pending = []
    skipped = []
    for entry in entries:
        source = Path(entry.get('source', str(ROOT / entry['path'])))
        require(source.is_file() and not source.is_symlink(), f'Missing/linked source: {source}')
        require(source.stat().st_size == entry['bytes'] and sha(source) == entry['sha256'],
                f'Pinned source SHA/size changed: {source}')
        target = safe_target(destination, entry['path'])
        if target.is_file() and target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256']:
            skipped.append(entry['path'])
        else:
            require(not (destination / 'DONE').exists(),
                    f'Completed backup contains a mismatch: {target}; refusing overwrite')
            pending.append((source, target, entry))
    data_bytes = sum(entry['bytes'] for entry in entries)
    needed = sum(entry['bytes'] for _, _, entry in pending)
    reserve = plan['self_chosen_reserve_bytes']
    metadata_allowance = 8 * 1024 * 1024
    free = shutil.disk_usage(destination.parent).free
    summary = {'mode': 'execute' if args.execute else 'dry_run_no_copy',
               'destination': str(destination), 'files': len(entries), 'total_bytes': data_bytes,
               'already_verified_files': len(skipped), 'bytes_to_copy': needed,
               'free_bytes_before': free, 'self_chosen_reserve_bytes': reserve,
               'reserve_is_user_requirement': False, 'metadata_allowance_bytes': metadata_allowance,
               'projected_free_after_data_bytes': free - needed,
               'space_sufficient': free >= needed + reserve + metadata_allowance,
               'training_state_verified': [{'run': p['run'], **p['resume_checkpoint']} for p in proofs]}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    require(summary['space_sufficient'],
            f'Insufficient SFS space: free={free}, need_copy={needed}, '
            f'self-chosen reserve20GiB={reserve}, metadata_allowance={metadata_allowance}; '
            'no other files were deleted or modified')
    if not args.execute:
        return
    lock_path = ROOT / 'device_locks/nr_push2_checkpoint_backup.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / 'DONE').exists():
            previous_done = json.loads((destination / 'DONE').read_text())
            previous_manifest = json.loads((destination / 'manifest.json').read_text())
            require(sha(destination / 'manifest.json') == previous_done['manifest_sha256'],
                    'Existing DONE does not match its manifest')
            require(previous_manifest['backup_plan_sha256'] == hashlib.sha256(plan_bytes).hexdigest(),
                    'Existing completed backup belongs to a different pinned plan')
            require(not pending, 'Existing completed backup still needs copies')
            print(json.dumps({'already_complete_and_reverified': str(destination),
                              'manifest_sha256': previous_done['manifest_sha256']}), flush=True)
            return
        remaining = needed
        for source, target, entry in pending:
            available = shutil.disk_usage(destination.parent).free
            require(available >= remaining + reserve + metadata_allowance,
                    f'SFS space changed before {entry["path"]}: free={available}, '
                    f'remaining={remaining}, self-chosen reserve={reserve}; no deletions performed')
            verified_stream_copy(source, target, entry)
            remaining -= entry['bytes']
            print(json.dumps({'copied_and_sha_verified': entry['path'], 'bytes': entry['bytes'],
                              'remaining_bytes': remaining}), flush=True)
        # Completion metadata is written only after every destination SHA passes.
        for entry in entries:
            target = safe_target(destination, entry['path'])
            require(target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'],
                    f'Final destination verification failed: {entry["path"]}')
        final_proofs = [run_proof(name) for name in plan['runs']]
        require(shutil.disk_usage(destination.parent).free >= reserve + metadata_allowance,
                'Self-chosen reserve no longer available after copies; no DONE written')
        manifest = {'schema': 'nr_push2_checkpoint_backup_v1', 'status': 'complete',
                    'source_root': str(ROOT), 'destination': str(destination),
                    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'scope': 'stage2 checkpoint/restart and frozen small source evidence only',
                    'final_quality_and_render_artifacts_pending_separate_finalize': True,
                    'parent_reference': plan['parent_reference'], 'run_proofs': final_proofs,
                    'files': [{'path': e['path'], 'bytes': e['bytes'], 'sha256': e['sha256']} for e in entries],
                    'count': len(entries), 'bytes': data_bytes,
                    'self_chosen_reserve_bytes': reserve, 'reserve_is_user_requirement': False,
                    'excluded': plan['excluded'],
                    'copy_method': 'direct stream to SFS final paths, full SHA256 source/readback verification; no copy2/chmod/rename',
                    'backup_script_sha256': plan['script_sha256'],
                    'backup_plan_sha256': hashlib.sha256(plan_bytes).hexdigest()}
        manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
        direct_write(destination / 'manifest.json', manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        done = {'status': 'complete', 'scope': 'stage2_checkpoint_backup_only',
                'files': len(entries), 'bytes': data_bytes, 'manifest_sha256': manifest_sha,
                'destination': str(destination), 'self_chosen_reserve_bytes': reserve,
                'finalize_pending': True}
        direct_write(destination / 'DONE', (json.dumps(done, indent=2) + '\n').encode())
        print(json.dumps({'backup_complete': done, 'free_bytes_after': shutil.disk_usage(destination.parent).free}), flush=True)


if __name__ == '__main__':
    main()
