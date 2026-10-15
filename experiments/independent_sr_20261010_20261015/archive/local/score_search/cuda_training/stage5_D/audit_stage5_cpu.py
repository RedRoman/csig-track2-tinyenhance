"""Plan-driven, CPU-only audit of a new NR stage from sealed parent7250.

The separate immutable plan must declare parent/anchor SHA256 and branch loss
weights before execution. Completed-run gates precede checkpoint reads. Generic
tensor helpers are imported from a SHA-pinned old CPU audit; its stage3-specific
entrypoint, run assertions, parent6250 assumptions and model code are never run.
No model is constructed and no CUDA RNG setter is called.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import traceback

HELPER_SHA = 'deb0c299adca7dd82dc8ff91e9584ea8d7d51d5e075d6e8dca8c9b25080bc2b9'
DEFAULT_HELPER = '/cache/independent_sr_cuda_20261013/audits/push2/training_stage3/audit_stage3_parameters_cpu.py'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def replay_crop(parent, protocol):
    from PIL import Image
    rng = random.Random(0)
    rng.setstate(parent['python_rng'])
    case = rng.randint(1, 100)
    record = protocol['inputs'][case - 1]
    need(record['case'] == case, 'Parent input order changed')
    with Image.open(record['path']) as image:
        width, height = image.size
    return case, [rng.randrange(width - 511), rng.randrange(height - 511)]


def snapshot_gate(run, branch, parent_plan, local_step):
    state = read(run / 'status.json')
    steps = local_step
    end_global = parent_plan['global_step'] + steps
    end_adam = parent_plan['optimizer_step'] + steps
    names = {'global_step': end_global, 'local_step': steps, 'optimizer_step': end_adam}
    resume = run / branch['milestone_resume_pattern'].format(**names)
    weight = run / f'step{end_global}.pth'
    need(resume.is_file() and weight.is_file(), f'Immutable milestone snapshot missing: {run}/{steps}')
    if branch.get('milestone_marker_pattern'):
        marker_path = run / branch['milestone_marker_pattern'].format(**names)
        need(marker_path.is_file(), f'Milestone marker: {run}/{steps}')
        marker = read(marker_path)
        need((marker['global_step'], marker['local_step'], marker['optimizer_step']) ==
             (end_global, steps, end_adam), f'Milestone marker step metadata: {run}/{steps}')
        need(Path(marker['raw_checkpoint']).resolve() == weight.resolve() and
             Path(marker['resume_checkpoint']).resolve() == resume.resolve(), f'Milestone marker paths: {run}/{steps}')
        stat = resume.stat()
        need((stat.st_size, stat.st_mtime_ns, stat.st_ino) ==
             (marker['resume_bytes'], marker['resume_mtime_ns'], marker['resume_inode']),
             f'Milestone immutable stat changed: {run}/{steps}')
        need(marker['latest_same_inode_at_save'] is True and marker['exact_FP32_LoRA_Adam_and_all4_RNG'] is True,
             f'Milestone recorded exact resume proof: {run}/{steps}')
    if steps == branch['stage_steps']:
        need(state['stage'] == 'complete', f'Run still active at terminal audit: {run}')
        need(state['completed_local_steps'] == steps, f'Local step completion: {run}')
        need(state['global_step'] == end_global, f'Global step completion: {run}')
        need(state['optimizer_step'] == end_adam, f'Adam step completion: {run}')
        need((run / f'training_complete_step{end_global}').is_file(), f'Completion marker: {run}')
    return resume, weight


def audit_log(run, branch, pp, expected_crop, steps):
    raw_lines = [line for line in (run / 'train.jsonl').read_text().splitlines(keepends=True) if line.strip()]
    rows = [json.loads(line) for line in raw_lines]
    need(len(rows) >= steps, f'Expected at least {steps} updates in log: {run}')
    rows = rows[:steps]
    for i, row in enumerate(rows, 1):
        need((row['local_step'], row['global_step'], row['optimizer_step']) ==
             (i, pp['global_step'] + i, pp['optimizer_step'] + i), f'Log step sequence: {run}/{i}')
        for field in ('loss', 'fidelity', 'reward', 'gradient_norm'):
            need(math.isfinite(row[field]), f'Nonfinite {field}: {run}/{i}')
        need(row['fidelity'] >= -1e-8, f'Negative fidelity: {run}/{i}')
        need(row['gradient_norm'] > 0 and row['nonzero_lora_gradients'] == 514,
             f'Not all 514 LoRA gradients nonzero: {run}/{i}')
        need(row['finite_lora_gradients'] == 514 and row['lr'] == branch['lr']
             and row['teacher'] == branch['teacher'], f'Gradient/lr/teacher log metadata: {run}/{i}')
        scores = row['scores']
        need(set(scores) == {'clipiqa', 'maniqa', 'musiq'}, f'Training reward metric set: {run}/{i}')
        need(all(math.isfinite(x) for x in scores.values()), f'Nonfinite training reward metric: {run}/{i}')
        expected_reward = (branch['musiq_weight'] * scores['musiq'] + scores['maniqa'] +
                           branch['clip_weight'] * scores['clipiqa'])
        need(abs(row['reward'] - expected_reward) < 5e-6, f'Reward identity: {run}/{i}')
        expected_loss = row['fidelity'] - branch['reward_weight'] * row['reward']
        need(abs(row['loss'] - expected_loss) < 5e-7, f'Loss identity: {run}/{i}')
    first = rows[0]
    need((first['case'], first['crop_xy']) == expected_crop, f'Parent RNG first crop: {run}')
    zero_expected = branch['anchor']['global_step'] == pp['global_step']
    if zero_expected:
        need(abs(first['fidelity']) <= 1e-6, f'Re-anchored initial fidelity should be zero: {run}')
    else:
        need(math.isfinite(first['fidelity']), f'Continuation first fidelity invalid: {run}')
    return rows, {'updates': len(rows), 'sequential_local_global_Adam_steps': True,
                  'audited_log_prefix_SHA256': hashlib.sha256(''.join(raw_lines[:steps]).encode()).hexdigest(),
                  'paired_case_crop_sequence_SHA256': identity([(r['case'], r['crop_xy']) for r in rows]),
                  'all_514_LoRA_gradients_nonzero_and_finite': True,
                  'training_reward_and_loss_recomputed_from_logged_components': True,
                  'first_case_crop_matches_parent_Python_RNG': True,
                  'initial_fidelity': first['fidelity'],
                  'initial_fidelity_expected_zero': zero_expected,
                  'min_fidelity': min(r['fidelity'] for r in rows),
                  'max_fidelity': max(r['fidelity'] for r in rows),
                  'last_fidelity': rows[-1]['fidelity']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--helper', type=Path, default=Path(DEFAULT_HELPER))
    ap.add_argument('--local-step', type=int, help='Audit this immutable milestone; default terminal stage step')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = {'schema': 'cuda_nr_stage5_plan_driven_CPU_audit_v1',
              'audited_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'CPU_only': True, 'branches': []}
    torch = None
    try:
        plan = read(args.plan)
        report['audit_plan_sha256'] = digest(args.plan)
        need(plan['schema'] == 'cuda_nr_stage5_independent_audit_plan_v1', 'Audit plan schema')
        root = Path(plan['root']).resolve()
        pp = plan['parent']
        need(pp['global_step'] == 7250 and pp['optimizer_step'] == 1250,
             'This new audit requires explicit parent7250/Adam1250 plan')
        branches = plan['branches']
        need(branches and len({b['run'] for b in branches}) == len(branches), 'Unique branch runs required')
        need(all(1 <= b['stage_steps'] <= 500 for b in branches), 'Short stage bound 1..500')
        for b in branches:
            run = root / 'adaptation/runs' / b['run']
            need(run.parent == root / 'adaptation/runs', 'Unsafe run path')
            local_step = args.local_step or b['stage_steps']
            need(1 <= local_step <= b['stage_steps'], 'Requested audit step out of stage bounds')
            snapshot_gate(run, b, pp, local_step)
        need(digest(args.helper) == HELPER_SHA, 'Frozen generic CPU tensor helper SHA')
        spec = importlib.util.spec_from_file_location('_frozen_tensor_audit_helpers', args.helper)
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        import torch as torch_module
        import numpy as np
        torch = torch_module
        torch.set_num_threads(2)
        need(not torch.cuda.is_initialized(), 'CUDA initialized before CPU audit')
        parent_dir = root / 'adaptation/runs' / pp['run']
        parent_resume_name = pp.get('resume_filename', 'resume_latest.pt')
        need(Path(parent_resume_name).name == parent_resume_name, 'Safe sealed parent resume filename required')
        parent_resume_path = parent_dir / parent_resume_name
        parent_checkpoint_name = pp['checkpoint_filename']
        need(Path(parent_checkpoint_name).name == parent_checkpoint_name, 'Safe sealed parent checkpoint filename required')
        parent_checkpoint_path = parent_dir / parent_checkpoint_name
        parent_tokens = {str(p): helpers.stat_token(p) for p in
                         (parent_resume_path, parent_checkpoint_path, parent_dir / 'config.json')}
        report['parent_files'] = {
            'resume': helpers.file_record(parent_resume_path, pp['resume_sha256']),
            'checkpoint': helpers.file_record(parent_checkpoint_path, pp['checkpoint_sha256']),
            'config': helpers.file_record(parent_dir / 'config.json', pp['config_sha256'])}
        cfg = read(parent_dir / 'config.json')
        need(identity(cfg['protocol']) == cfg['protocol_id'] == pp['protocol_id'], 'Sealed parent protocol ID')
        parent = torch.load(parent_resume_path, map_location='cpu', weights_only=False, mmap=True)
        need(parent['protocol_id'] == pp['protocol_id'] and parent['global_step'] == pp['global_step']
             and parent['local_step'] == pp['local_step'], 'Sealed parent resume metadata')
        report['parent_LoRA'] = helpers.validate_lora(parent['lora'], torch, context='sealed_parent7250')
        raw_parent = torch.load(parent_checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
        helpers.validate_lora(raw_parent, torch, parent['lora'], context='sealed_parent7250_raw')
        need(list(raw_parent) == list(parent['lora']) and all(torch.equal(v, parent['lora'][k]) for k, v in raw_parent.items()),
             'Sealed parent standalone raw differs from immutable resume FP32 LoRA')
        report['parent_raw_exact_equal_immutable_resume_FP32'] = True
        del raw_parent
        parent_adam = helpers.audit_adam(parent['optimizer'], parent['lora'], pp['optimizer_step'], torch, 'parent7250')
        report['parent_Adam'] = parent_adam
        expected_crop = replay_crop(parent, cfg['protocol'])
        report['independent_first_case_crop'] = {'case': expected_crop[0], 'crop_xy': expected_crop[1]}
        paired_logs = {}
        for branch in branches:
            name = branch['run']; run = root / 'adaptation/runs' / name
            CLI_record = helpers.file_record(branch['CLI_config_path'], branch['CLI_config_sha256'])
            local_step = args.local_step or branch['stage_steps']
            target_global = pp['global_step'] + local_step
            target_adam = pp['optimizer_step'] + local_step
            resume_path, weight_path = snapshot_gate(run, branch, pp, local_step)
            tracked = [run / 'config.json', run / 'parent_migration.json', resume_path, weight_path]
            before = {str(p): helpers.stat_token(p) for p in tracked}
            config = read(run / 'config.json'); protocol = config['protocol']
            need(identity(protocol) == config['protocol_id'], f'Branch protocol digest: {name}')
            need(config['target_stage_steps'] == branch['stage_steps'] and
                 config['target_global_step'] == pp['global_step'] + branch['stage_steps'] and
                 config['target_optimizer_step'] == pp['optimizer_step'] + branch['stage_steps'] and
                 config['save_every'] == 250, f'Configured stage targets: {name}')
            for field, expected in branch['expected_protocol_fields'].items():
                value = protocol
                for part in field.split('.'):
                    value = value[part]
                need(value == expected, f'Expected sealed protocol {name}/{field}')
            need(len(branch['expected_protocol_fields']) >= 12, f'Independent protocol expectations incomplete: {name}')
            for field in ('lr', 'reward_weight', 'gradient_anchor', 'musiq_weight', 'clip_weight'):
                need(protocol[field] == branch[field], f'Branch hyperparameter {name}/{field}')
            for field in ('coeff_t', 'model_t', 'rank', 'batch', 'patch', 'maniqa_train_samples',
                          'inputs', 'metric_weights_sha256', 'base_provenance', 'versions', 'compile', 'fuse'):
                need(protocol[field] == cfg['protocol'][field], f'Unchanged inherited protocol {name}/{field}')
            source_records = {p: helpers.file_record(p, expected)
                              for p, expected in protocol['source_sha256'].items()}
            need(protocol['source_sha256'].get(plan['trainer_path']) == plan['trainer_sha256'],
                 f'Frozen new trainer not pinned in protocol: {name}')
            for p, expected in cfg['protocol']['source_sha256'].items():
                need(protocol['source_sha256'].get(p) == expected, f'Inherited frozen source missing: {name}/{p}')
            anchor = branch['anchor']; anchor_dir = root / 'adaptation/runs' / anchor['run']
            anchor_record = helpers.file_record(anchor_dir / 'resume_latest.pt', anchor['resume_sha256'])
            need(Path(protocol['anchor_checkpoint']).resolve() == (anchor_dir / 'resume_latest.pt').resolve()
                 and protocol['anchor_sha256'] == anchor['resume_sha256'], f'Effective teacher anchor: {name}')
            migration = read(run / 'parent_migration.json')
            need(migration['protocol_id'] == config['protocol_id'] and
                 migration['parent_lineage'] == protocol['parent_lineage'] and
                 migration['FP32_LoRA_exact_equal_before_update'] is True and
                 migration['FP32_LoRA_tensors'] == 514, f'Exact FP32 migration proof: {name}')
            need(migration['FP32_Adam_moments_exact_equal_before_update'] is True and
                 migration['Adam_parameter_states'] == 514 and
                 migration['loaded_optimizer_group_lrs'] == [cfg['protocol']['lr']] and
                 migration['applied_optimizer_group_lrs'] == [branch['lr']] and
                 migration['lr_overwrite_after_load'] is True, f'Exact Adam migration and applied LR proof: {name}')
            need(migration['parent_raw_exact_equal_immutable_resume_FP32'] is True and
                 migration['parent_checkpoint_sha256'] == pp['checkpoint_sha256'],
                 f'Parent raw versus immutable resume migration proof: {name}')
            need(migration[branch['teacher_proof_key']] is True and migration['teacher_frozen'] is True,
                 f'Teacher FP32/frozen proof: {name}')
            final = torch.load(weight_path, map_location='cpu', weights_only=True, mmap=True)
            saved = torch.load(resume_path, map_location='cpu', weights_only=False, mmap=True)
            structure = helpers.validate_lora(final, torch, parent['lora'], context=name)
            need(saved['protocol_id'] == config['protocol_id'] and saved['global_step'] == target_global
                 and saved['local_step'] == local_step and saved['optimizer_step'] == target_adam,
                 f'Resume step/protocol metadata: {name}')
            need(saved['parent_lineage'] == protocol['parent_lineage'], f'Resume parent lineage: {name}')
            need(list(saved['lora']) == list(final) == list(parent['lora']), f'Exact LoRA key order: {name}')
            for key, value in final.items():
                need(saved['lora'][key].dtype == torch.float32 and torch.equal(value, saved['lora'][key]),
                     f'Final weight versus resume differs: {name}/{key}')
            delta = helpers.fp32_delta(final, parent['lora'], torch)
            adam = helpers.audit_adam(saved['optimizer'], final, target_adam, torch, name)
            need(adam['parameter_group_options'] == parent_adam['parameter_group_options'], f'Inherited Adam options: {name}')
            rng = helpers.rng_evidence(saved, torch, np)
            rows, log_audit = audit_log(run, branch, pp, expected_crop, local_step)
            paired_logs[name] = [(r['case'], r['crop_xy']) for r in rows]
            records = {p.name: helpers.file_record(p) for p in tracked}
            marker_path = run / branch['milestone_marker_pattern'].format(
                global_step=target_global, local_step=local_step, optimizer_step=target_adam)
            marker = read(marker_path)
            need(records[weight_path.name]['sha256'] == marker['raw_checkpoint_sha256'] and
                 records[resume_path.name]['sha256'] == marker['resume_checkpoint_sha256'],
                 f'Milestone completion-marker SHA mismatch: {name}')
            records[marker_path.name] = helpers.file_record(marker_path)
            need(all(helpers.stat_token(p) == before[str(p)] for p in tracked), f'Files changed during audit: {name}')
            report['branches'].append({'run': name, 'status': 'PASS', 'protocol_id': config['protocol_id'],
                'audited_stage_local_step': local_step, 'audited_global_step': target_global,
                'audited_Adam_step': target_adam, 'run_stage_target_local_step': branch['stage_steps'],
                'LoRA': structure, 'FP32_delta_vs_parent7250': delta,
                'launch_CLI_config_record': CLI_record,
                'final_weight_exact_equal_resume_FP32_LoRA': True, 'Adam': adam, 'final_rng': rng,
                'training_log': log_audit, 'file_records': records, 'source_records': source_records,
                'teacher_anchor_record': anchor_record,
                'teacher_proof_scope': 'Saved migration proof + SHA-frozen trainer source; no teacher model reconstructed by CPU audit.'})
            del saved, final
        first_name = branches[0]['run']
        paired = {name: rows == paired_logs[first_name] for name, rows in paired_logs.items()}
        report['paired_case_crop_sequences_equal_to_first_branch'] = paired
        need(all(paired.values()), 'Branches used different case/crop sequences')
        reference_rng = report['branches'][0]['final_rng']['state_sha256']
        report['paired_final4_RNG_equal'] = {
            row['run']: {key: value == reference_rng[key] for key, value in row['final_rng']['state_sha256'].items()}
            for row in report['branches']}
        need(all(all(v.values()) for v in report['paired_final4_RNG_equal'].values()), 'Final4 branch RNG differs')
        need(all(helpers.stat_token(p) == token for p, token in parent_tokens.items()),
             'Sealed parent files changed during CPU audit')
        report['parent_size_and_mtime_unchanged_during_audit'] = True
        need(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
        report['CUDA_context_initialized'] = False
        report['status'] = 'PASS'
        code = 0
    except Exception as exc:
        report.update(status='FAIL', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        report['CUDA_context_initialized'] = torch.cuda.is_initialized() if torch is not None else False
        code = 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
