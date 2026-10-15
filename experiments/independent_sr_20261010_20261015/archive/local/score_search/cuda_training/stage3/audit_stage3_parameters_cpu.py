"""Deferred CPU-only parameter/Adam/RNG audit of the two completed stage3 parent-anchor runs.

This script is prepared locally and must run only after both COMPLETE events.
It refuses incomplete status/markers before hashing or loading any checkpoint.
Intended remote invocation: CUDA_VISIBLE_DEVICES='' ROOT/env/bin/python -B -
with this file supplied on stdin. The only output is one JSON document on stdout.
No remote files are written, models constructed, or CUDA RNG states restored.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
import traceback

ROOT_DEFAULT = '/cache/independent_sr_cuda_20261013'
PARENT_RUN = 'cuda_nr_s6000_ct175_conservative_pilot100'
PARENT_PROTOCOL = 'fa5ea19179561aac206b26f7dd3596406f1ad376dccd38f49821146623b98d52'
PARENT_RESUME_SHA = '3dcf896028f41c7c4c6e15074cc69c4cdb2ff7679e6756f5ee57a9a657f61c1f'
PARENT_CONFIG_SHA = '2a459e95e248d1b502b651753b9598ce7adfb92c24e48863458265458ea1ca5a'
V1_SOURCE_SHA = '0998967aa6052ec9a5090219e17cb73c05468b06d7acc1eae27108c1c412fb4b'
V2_SOURCE_SHA = '9ac86a8d78ba0fc085b60278a460ecaf1ffef74d0dd4e21c4b5c5d84063c4651'
V3_SOURCE_SHA = '539844657210938295aa55eb71a2620bad40b974a9ae48ad7238ae7d70fa7c2c'
RUNS = (
    ('cuda_nr_stage3_s6250_anchor6250_m1_250', 1.0),
    ('cuda_nr_stage3_s6250_anchor6250_m15_250', 1.5),
)
KEYS = 514
ELEMENTS = 259474432
CHUNK = 1048576
TZ = datetime.timezone(datetime.timedelta(hours=8))


class AuditFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def identity(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def stat_token(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns


def file_record(path, expected_sha=None):
    path = Path(path)
    before = stat_token(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    require(stat_token(path) == before, f'File changed while hashing: {path}')
    actual = h.hexdigest()
    if expected_sha is not None:
        require(actual == expected_sha, f'SHA mismatch: {path}')
    return {
        'path': str(path), 'bytes': before[0], 'sha256': actual,
        'mtime_bjt': datetime.datetime.fromtimestamp(before[1] / 1e9, TZ).isoformat(),
    }


def require_complete(run):
    """Small-file guard; deliberately precedes all checkpoint reads."""
    state = load_json(run / 'status.json')
    require(state.get('stage') == 'complete', f'Stage not complete: {run}')
    require(state.get('completed_local_steps') == 250, f'Expected stage local250: {run}')
    require(state.get('global_step') == 6500, f'Expected global6500: {run}')
    require(state.get('optimizer_step') == 500, f'Expected Adam500 metadata: {run}')
    marker = run / 'training_complete_step6500'
    require(marker.is_file(), f'Completion marker missing: {marker}')
    return state


def chunks(value):
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), CHUNK):
        yield flat[start:start + CHUNK]


def require_finite_cpu(value, torch, context):
    require(isinstance(value, torch.Tensor), f'Not a tensor: {context}')
    require(value.device.type == 'cpu', f'Non-CPU tensor: {context}')
    for block in chunks(value):
        require(bool(torch.isfinite(block).all()), f'Nonfinite tensor: {context}')


def validate_lora(state, torch, reference=None, context='LoRA'):
    require(len(state) == KEYS, f'{context}: expected514 keys')
    require(sum(x.numel() for x in state.values()) == ELEMENTS,
            f'{context}: element count mismatch')
    if reference is not None:
        require(set(state) == set(reference), f'{context}: key set differs')
    for name, tensor in state.items():
        require(tensor.dtype == torch.float32, f'{context}: non-FP32 {name}')
        require_finite_cpu(tensor, torch, f'{context}/{name}')
        if reference is not None:
            require(tensor.shape == reference[name].shape, f'{context}: shape differs {name}')
    return {'keys': KEYS, 'elements': ELEMENTS, 'dtype': 'torch.float32',
            'device': 'cpu', 'all_finite': True}


def fp32_delta(current, parent, torch):
    """All elements, not serialization hashes; bounded-memory FP64 reductions."""
    total = changed = tensors_changed = 0
    abs_sum = square_sum = maximum = 0.0
    per_tensor = []
    for name, value in current.items():
        previous = parent[name].reshape(-1)
        flat = value.reshape(-1)
        local_changed = 0
        local_max = local_ss = 0.0
        for start in range(0, flat.numel(), CHUNK):
            d = flat[start:start + CHUNK] - previous[start:start + CHUNK]
            dd = d.double()
            local_changed += int((d != 0).sum().item())
            local_max = max(local_max, float(d.abs().max().item()))
            ss = float(dd.square().sum().item())
            local_ss += ss
            square_sum += ss
            abs_sum += float(dd.abs().sum().item())
        total += flat.numel()
        changed += local_changed
        tensors_changed += int(local_changed > 0)
        maximum = max(maximum, local_max)
        per_tensor.append({'key': name, 'elements': flat.numel(),
                           'changed_elements': local_changed, 'max_abs': local_max,
                           'l2': math.sqrt(local_ss)})
    require(total == ELEMENTS and tensors_changed == KEYS and changed > 0,
            'Expected real FP32 update in all514 LoRA tensors')
    return {
        'elements': total, 'changed_elements': changed, 'changed_fraction': changed / total,
        'tensors_changed': tensors_changed, 'max_abs': maximum,
        'l2': math.sqrt(square_sum), 'mean_abs': abs_sum / total,
        'method': 'FP32 elementwise subtraction; FP64 sum(abs) and sum(square); every tensor/element checked',
        'largest10_tensors_by_max_abs': sorted(per_tensor, key=lambda x: x['max_abs'], reverse=True)[:10],
    }


def audit_adam(optimizer, lora, expected_step, torch, context):
    require(set(optimizer) >= {'state', 'param_groups'}, f'{context}: optimizer schema')
    states, groups = optimizer['state'], optimizer['param_groups']
    require(len(groups) == 1, f'{context}: expected one parameter group')
    ids = [pid for group in groups for pid in group['params']]
    require(len(ids) == len(set(ids)) == len(states) == KEYS,
            f'{context}: Adam514 IDs/count mismatch')
    require(set(ids) == set(states), f'{context}: state IDs do not cover groups exactly')
    options = {key: value for key, value in groups[0].items() if key != 'params'}
    require(options.get('lr') == 2e-7 and options.get('weight_decay') == 0,
            f'{context}: LR/weight_decay changed')
    require(tuple(options.get('betas', ())) == (0.9, 0.999) and options.get('eps') == 1e-8,
            f'{context}: Adam betas/eps changed')
    require(options.get('foreach') is False and options.get('amsgrad') is False,
            f'{context}: unexpected optimizer implementation/options')
    field_sets, dtypes, steps = set(), set(), set()
    moment_elements = 0
    all_nonzero = True
    for pid, (name, parameter) in zip(ids, lora.items()):
        st = states[pid]
        field_sets.add(tuple(sorted(st)))
        require({'step', 'exp_avg', 'exp_avg_sq'} <= set(st), f'{context}/{name}: missing state')
        step = st['step']
        require_finite_cpu(step, torch, f'{context}/{name}/step')
        require(step.numel() == 1 and float(step.item()) == expected_step,
                f'{context}/{name}: Adam step must be {expected_step}')
        steps.add(int(step.item()))
        for field in ('exp_avg', 'exp_avg_sq'):
            value = st[field]
            require(value.shape == parameter.shape and value.dtype == torch.float32,
                    f'{context}/{name}/{field}: wrong shape/dtype')
            require_finite_cpu(value, torch, f'{context}/{name}/{field}')
            dtypes.add(str(value.dtype))
        require(bool((st['exp_avg_sq'] >= 0).all()), f'{context}/{name}: negative second moment')
        all_nonzero = all_nonzero and bool((st['exp_avg'] != 0).any())
        moment_elements += st['exp_avg'].numel()
    require(moment_elements == ELEMENTS, f'{context}: moment element count')
    return {
        'state_entries': KEYS, 'parameter_ids_complete_unique': True,
        'shape_to_parameter_order_checked': True, 'all_steps': sorted(steps),
        'all_moments_FP32_CPU_finite': True, 'moment_elements_each': moment_elements,
        'exp_avg_sq_nonnegative': True, 'every_exp_avg_tensor_has_nonzero_values': all_nonzero,
        'state_field_sets': [list(x) for x in sorted(field_sets)],
        'moment_dtypes': sorted(dtypes), 'parameter_group_options': options,
    }


def rng_evidence(saved, torch, np):
    """Content hashes only; no torch.cuda RNG setter and no global RNG restore."""
    required = {'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'}
    require(required <= set(saved), 'Missing saved RNG state')
    py = saved['python_rng']
    independent_python = random.Random(0)
    independent_python.setstate(py)
    n = saved['numpy_rng']
    require(isinstance(n, tuple) and len(n) == 5, 'NumPy RNG schema mismatch')
    independent_numpy = np.random.RandomState(0)
    independent_numpy.set_state(n)
    cpu, cuda = saved['torch_rng'], saved['cuda_rng']
    for name, value in (('torch_rng', cpu), ('cuda_rng', cuda)):
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
                f'Invalid saved {name} byte vector')
    torch.Generator(device='cpu').set_state(cpu)
    digests = {
        'python': hashlib.sha256(json.dumps(py).encode()).hexdigest(),
        'numpy': hashlib.sha256(n[1].tobytes() + repr((n[0], n[2], n[3], n[4])).encode()).hexdigest(),
        'cpu': hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        'cuda': hashlib.sha256(cuda.numpy().tobytes()).hexdigest(),
    }
    return {
        'state_sha256': digests, 'python_and_numpy_state_loadable_in_independent_instances': True,
        'torch_cpu_state_loadable_in_independent_CPU_generator': True,
        'torch_cpu_rng_bytes': cpu.numel(), 'cuda_rng_bytes': cuda.numel(),
        'cuda_rng_loaded_as_CPU_bytes_only_never_restored': True,
        'hash_encoding': 'Python json.dumps(state); NumPy key raw bytes plus repr of algorithm/position/Gaussian cache; Torch raw uint8 bytes',
    }


def replay_first_crop(parent, parent_protocol):
    """Only the three Python draws before first metrics; never replay all250 crops."""
    from PIL import Image
    replay = random.Random(0)
    replay.setstate(parent['python_rng'])
    case = replay.randint(1, 100)
    record = parent_protocol['inputs'][case - 1]
    require(record['case'] == case, 'Parent input case ordering changed')
    with Image.open(record['path']) as image:
        width, height = image.size
    require(min(width, height) >= 512, 'Replay source is smaller than512')
    return {'case': case, 'crop_xy': [replay.randrange(width - 511), replay.randrange(height - 511)],
            'source_path': record['path'], 'source_dimensions_wh': [width, height],
            'scope': 'First stage case/crop before first metrics only; no claim about all250 Python/NumPy draws'}


def first_logged_crop(run, expected):
    with (run / 'train.jsonl').open() as stream:
        row = next((json.loads(line) for line in stream if line.strip()), None)
    require(row is not None, f'No first stage training row: {run}')
    require(row['local_step'] == 1 and row['global_step'] == 6251 and row['optimizer_step'] == 251,
            f'First logged stage step semantics differ: {run}')
    require(row['case'] == expected['case'] and row['crop_xy'] == expected['crop_xy'],
            f'First Python RNG case/crop differs from parent replay: {run}')
    fidelity = float(row['fidelity'])
    require(math.isfinite(fidelity) and abs(fidelity) <= 1e-6,
            f'Fresh-stage parent6250 fidelity is not numerically zero: {run}')
    return {'local_step': row['local_step'], 'global_step': row['global_step'],
            'optimizer_step': row['optimizer_step'], 'case': row['case'], 'crop_xy': row['crop_xy'],
            'matches_independent_parent_Python_RNG_replay': True,
            'initial_fidelity': fidelity, 'initial_fidelity_tolerance': 1e-6,
            'initial_fidelity_abs_within_tolerance': True}


def audit_run(root, run_name, musiq_weight, parent, parent_protocol, parent_adam, expected_crop, torch, np):
    run = root / 'adaptation/runs' / run_name
    tracked = [run / name for name in ('step6500.pth', 'resume_latest.pt', 'config.json', 'parent_migration.json')]
    start_tokens = {str(path): stat_token(path) for path in tracked}
    cfg = load_json(run / 'config.json')
    protocol = cfg['protocol']
    require(identity(protocol) == cfg['protocol_id'], f'New protocol digest mismatch: {run_name}')
    require(protocol['schema'] == 'cuda_raw_hypir_nr_stage_v3_parentanchor', f'New protocol schema: {run_name}')
    require(protocol['nr_round_ste'] is False, f'V3 must disable STE: {run_name}')
    require(protocol['musiq_weight'] == musiq_weight, f'MUSIQ weight branch mismatch: {run_name}')
    require(protocol['stage_start_global_step'] == 6250 and protocol['parent_optimizer_step'] == 250,
            f'Protocol phase start mismatch: {run_name}')
    require(cfg['target_stage_steps'] == 250 and cfg['target_global_step'] == 6500
            and cfg['target_optimizer_step'] == 500, f'Config stage targets mismatch: {run_name}')
    lineage = protocol['parent_lineage']
    require(lineage['parent_resume_sha256'] == PARENT_RESUME_SHA
            and lineage['parent_config_sha256'] == PARENT_CONFIG_SHA
            and lineage['parent_protocol_id'] == PARENT_PROTOCOL, f'Parent seals differ: {run_name}')
    require(lineage['parent_local_step'] == 250 and lineage['parent_global_step'] == 6250
            and lineage['parent_optimizer_step'] == 250, f'Parent lineage step mismatch: {run_name}')
    parent_dir = root / 'adaptation/runs' / PARENT_RUN
    require(Path(lineage['parent_resume']).resolve() == (parent_dir / 'resume_latest.pt').resolve()
            and Path(lineage['parent_config']).resolve() == (parent_dir / 'config.json').resolve(),
            f'Unexpected parent lineage paths: {run_name}')
    for key in ('init_checkpoint', 'init_sha256', 'base_step',
                'lr', 'seed', 'coeff_t', 'model_t', 'rank', 'batch', 'patch', 'reward_weight',
                'clip_weight', 'gradient_anchor', 'maniqa_train_samples', 'inputs', 'metric_weights_sha256',
                'base_provenance', 'versions', 'compile', 'fuse'):
        require(protocol[key] == parent_protocol[key], f'Inherited setting changed: {run_name}/{key}')
    require(Path(protocol['anchor_checkpoint']).resolve() == (parent_dir / 'resume_latest.pt').resolve()
            and protocol['anchor_sha256'] == PARENT_RESUME_SHA,
            f'Effective teacher anchor must be sealed parent6250 resume: {run_name}')
    require(Path(protocol['anchor_initialization_checkpoint']).resolve()
            == Path(parent_protocol['anchor_checkpoint']).resolve()
            and protocol['anchor_initialization_sha256'] == parent_protocol['anchor_sha256'],
            f'Teacher initializer must remain original raw6000: {run_name}')
    require(protocol['teacher_FP32_exact_parent6250'] is True and protocol['teacher_frozen'] is True,
            f'Teacher protocol proof missing: {run_name}')
    require(protocol['fresh_stage_initial_fidelity_expected_zero'] is True
            and protocol['initial_fidelity_tolerance'] == 1e-6,
            f'Fresh-stage fidelity guard declaration mismatch: {run_name}')
    require(protocol['anchor_loading'] == 'all514 parent_resume.lora FP32 values copied exactly into raw6000-initialized teacher; not direct resume-file weight loading',
            f'Teacher loading declaration mismatch: {run_name}')
    require(protocol['loss'] == 'L1(parent6250)+.1*LPIPS(parent6250)+gradient_anchor*spatial_gradient_L1 - reward_weight*(musiq_weight*MUSIQ/100+MANIQA+clip_weight*CLIPIQA)',
            f'V3 loss declaration mismatch: {run_name}')
    require(lineage['musiq_weight'] == musiq_weight
            and lineage['objective_relation'] == 'new frozen6250 exact-FP32 fidelity anchor objective; inherited Adam/RNG warmstart',
            f'V3 objective lineage mismatch: {run_name}')
    sources = protocol['source_sha256']
    v1 = root / 'code/cuda_training/train_cuda_nr.py'
    v2 = root / 'code/cuda_training/train_cuda_nr_stage_v2.py'
    v3 = root / 'code/cuda_training/train_cuda_nr_stage_v3.py'
    helper = root / 'code/cuda_training/png_quantization_surrogate.py'
    require(sources.get(str(v1)) == V1_SOURCE_SHA and sources.get(str(v2)) == V2_SOURCE_SHA
            and sources.get(str(v3)) == V3_SOURCE_SHA,
            f'Frozen v1/v2/v3 source identity missing: {run_name}')
    require(sources.get(str(helper)) == protocol['quantization_helper_sha256'],
            f'Quantization helper missing from source identity: {run_name}')
    require(all(sources.get(path) == expected for path, expected in parent_protocol['source_sha256'].items()),
            f'Inherited parent source identity changed or omitted: {run_name}')
    require(set(sources) == set(parent_protocol['source_sha256']) | {str(v2), str(v3), str(helper)},
            f'Unexpected stage source graph: {run_name}')
    require(lineage['frozen_v1_source_sha256'] == V1_SOURCE_SHA
            and Path(lineage['frozen_v1_source']).resolve() == v1.resolve(),
            f'Lineage frozen helper identity mismatch: {run_name}')
    require(lineage['derived_v2_source_sha256'] == V2_SOURCE_SHA
            and Path(lineage['derived_v2_source']).resolve() == v2.resolve(),
            f'V3 derivation source lineage mismatch: {run_name}')
    source_records = {path: file_record(path, expected) for path, expected in sources.items()}
    migration = load_json(run / 'parent_migration.json')
    require(migration['protocol_id'] == cfg['protocol_id'] and migration['parent_lineage'] == lineage,
            f'Migration/lineage mismatch: {run_name}')
    require(migration['FP32_LoRA_exact_equal_before_update'] is True
            and migration['FP32_LoRA_tensors'] == KEYS, f'FP32 migration proof missing: {run_name}')
    require(migration['teacher_FP32_exact_parent6250'] is True
            and migration['teacher_FP32_tensors'] == KEYS and migration['teacher_frozen'] is True
            and migration['anchor_resume_sha256'] == PARENT_RESUME_SHA
            and migration['musiq_weight'] == musiq_weight, f'Teacher migration proof mismatch: {run_name}')
    final = torch.load(run / 'step6500.pth', map_location='cpu', weights_only=True, mmap=True)
    saved = torch.load(run / 'resume_latest.pt', map_location='cpu', weights_only=False, mmap=True)
    final_structure = validate_lora(final, torch, parent['lora'], f'{run_name}/step6500')
    required = {'protocol_id', 'lora', 'optimizer', 'local_step', 'global_step', 'optimizer_step',
                'parent_lineage', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'}
    require(required <= set(saved), f'Stage resume fields missing: {run_name}')
    require(saved['protocol_id'] == cfg['protocol_id'] and saved['parent_lineage'] == lineage,
            f'Stage resume protocol/lineage mismatch: {run_name}')
    require(saved['local_step'] == 250 and saved['global_step'] == 6500 and saved['optimizer_step'] == 500,
            f'Stage resume local/global/optimizer metadata mismatch: {run_name}')
    require(list(saved['lora']) == list(final), f'Saved/standalone LoRA key order differs: {run_name}')
    require(list(final) == list(parent['lora']), f'Parent/final parameter order differs: {run_name}')
    for name, value in final.items():
        other = saved['lora'][name]
        require(other.dtype == torch.float32 and other.device.type == 'cpu'
                and other.shape == value.shape and torch.equal(other, value),
                f'Resume LoRA is not exact step6500: {run_name}/{name}')
    delta = fp32_delta(final, parent['lora'], torch)
    adam = audit_adam(saved['optimizer'], final, 500, torch, run_name)
    require(adam['parameter_group_options'] == parent_adam['parameter_group_options'],
            f'Adam options changed versus sealed parent: {run_name}')
    rng = rng_evidence(saved, torch, np)
    first_crop = first_logged_crop(run, expected_crop)
    records = {path.name: file_record(path) for path in tracked}
    require(all(stat_token(path) == start_tokens[str(path)] for path in tracked),
            f'Completed files changed during CPU audit: {run_name}')
    require(not torch.cuda.is_initialized(), 'CUDA context initialized during run audit')
    return {
        'run': str(run), 'status': 'PASS', 'nr_round_ste': False, 'musiq_weight': musiq_weight,
        'recorded_teacher_initialization_proof': {
            'effective_anchor_resume_SHA256': PARENT_RESUME_SHA,
            'initializer_checkpoint': protocol['anchor_initialization_checkpoint'],
            'initializer_SHA256': protocol['anchor_initialization_sha256'],
            'FP32_exact_parent6250': True, 'FP32_LoRA_tensors': KEYS, 'frozen': True,
            'anchor_loading': protocol['anchor_loading'],
            'evidence_scope': 'Migration proof and frozen training source checked; no teacher network reconstructed by this CPU audit.'},
        'protocol_id': cfg['protocol_id'], 'parent_lineage': lineage,
        'standalone_step6500_structure': final_structure,
        'resume_LoRA_exact_equal_to_standalone_step6500': True,
        'real_FP32_delta_vs_sealed_parent6250_resume': delta,
        'resume_steps': {'stage_local': 250, 'global': 6500, 'Adam': 500},
        'Adam': adam, 'final_rng': rng, 'first_case_crop_parent_RNG_replay': first_crop,
        'checkpoint_and_protocol_records': records, 'source_records': source_records,
        'completed_file_size_and_mtime_unchanged_during_audit': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(ROOT_DEFAULT))
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = {'schema': 'cuda_nr_stage3_parameter_CPU_audit_v1',
              'audited_bjt': datetime.datetime.now(TZ).isoformat(), 'root': str(root),
              'CPU_only': True, 'CUDA_VISIBLE_DEVICES': '', 'results': [],
              'scope': 'Completed stage3 parent6250-anchor FP32/Adam/protocol/RNG integrity; not a model quality or full training-log audit.'}
    torch = None
    try:
        # Guard both runs before any checkpoint load/hash; this script is deferred.
        report['completion_gates'] = {name: require_complete(root / 'adaptation/runs' / name)
                                      for name, _ in RUNS}
        import torch as torch_module
        import numpy as np
        torch = torch_module
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), 'CUDA context exists at audit start')
        report['cuda_context_initialized_before'] = torch.cuda.is_initialized()
        report['torch_version'] = torch.__version__
        parent_dir = root / 'adaptation/runs' / PARENT_RUN
        parent_paths = [parent_dir / 'resume_latest.pt', parent_dir / 'config.json',
                        root / 'code/cuda_training/train_cuda_nr.py']
        initial_tokens = {str(path): stat_token(path) for path in parent_paths}
        report['sealed_parent_files'] = {
            'resume': file_record(parent_paths[0], PARENT_RESUME_SHA),
            'config': file_record(parent_paths[1], PARENT_CONFIG_SHA),
            'v1_source': file_record(parent_paths[2], V1_SOURCE_SHA),
        }
        parent_cfg = load_json(parent_paths[1]);pp = parent_cfg['protocol']
        require(identity(pp) == parent_cfg['protocol_id'] == PARENT_PROTOCOL, 'Parent protocol identity mismatch')
        parent = torch.load(parent_paths[0], map_location='cpu', weights_only=False, mmap=True)
        require(parent['protocol_id'] == PARENT_PROTOCOL and parent['local_step'] == 250
                and parent['global_step'] == 6250, 'Sealed parent resume step/protocol mismatch')
        report['parent_LoRA'] = validate_lora(parent['lora'], torch, context='sealed_parent6250')
        parent_adam = audit_adam(parent['optimizer'], parent['lora'], 250, torch, 'sealed_parent6250')
        report['parent_Adam'] = parent_adam
        report['parent_rng'] = rng_evidence(parent, torch, np)
        expected_crop = replay_first_crop(parent, pp)
        report['independent_first_crop_replay'] = expected_crop
        for name, musiq_weight in RUNS:
            report['results'].append(audit_run(root, name, musiq_weight, parent, pp, parent_adam,
                                               expected_crop, torch, np))
        a, b = [item['final_rng']['state_sha256'] for item in report['results']]
        equal = {key: a[key] == b[key] for key in ('python', 'numpy', 'cpu', 'cuda')}
        report['cross_branch_final4_rng_comparison'] = {
            'equal_by_content_hash': equal, 'all4_equal': all(equal.values()),
            'status': 'EQUAL' if all(equal.values()) else 'DIFFERENT',
            'interpretation': 'Content-state comparison only; no CUDA state was restored. A difference is surfaced separately from parameter-integrity PASS.'}
        require(all(equal.values()), 'The two stage3 branches ended with different saved RNG content states')
        require(all(stat_token(path) == initial_tokens[str(path)] for path in parent_paths),
                'Parent file size/mtime changed during audit')
        require(not torch.cuda.is_initialized(), 'CUDA context initialized at audit end')
        report['cuda_context_initialized_after'] = torch.cuda.is_initialized()
        report['parent_files_unchanged_during_audit'] = True
        report['load_policy'] = 'torch.load(map_location=cpu,mmap=True); weights_only=True for standalone .pth and False for sealed trusted resume; no model construction/forward.'
        report['status'] = 'PASS'
        exit_code = 0
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        report['cuda_context_initialized_after'] = torch.cuda.is_initialized() if torch is not None else False
        report['torch_imported'] = torch is not None
        exit_code = 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
