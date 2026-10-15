"""CPU progress snapshot for the third CUDA score-improvement experiment."""
from pathlib import Path
import datetime
import json

ROOT = Path('/cache/independent_sr_cuda_20261013')
RUNS = {
    'A': 'cuda_nr_stage4_s6500_teacher6250_cont750',
    'B': 'cuda_nr_stage4_s6500_teacher6500_cont750',
    'C': 'cuda_nr_stage4_s6500_teacher6500_nrstrong750',
}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


training = {}
for branch, run in RUNS.items():
    directory = ROOT / 'adaptation/runs' / run
    status = read(directory / 'status.json') or {'stage': 'initializing'}
    entry = {k: status[k] for k in ['stage', 'completed_local_steps', 'total', 'global_step', 'optimizer_step'] if k in status}
    row = status.get('row', {})
    entry.update({k: row[k] for k in ['global_step', 'optimizer_step', 'loss', 'fidelity', 'reward', 'nonzero_lora_gradients', 'finite_lora_gradients'] if k in row})
    entry['run'] = run
    entry['milestones'] = {}
    for step in (6750, 7000, 7250):
        marker = read(directory / f'checkpoint_step{step}.json')
        if marker:
            entry['milestones'][str(step)] = marker
    preflight = read(directory / 'gradient_preflight.json')
    if preflight:
        entry['gradient_preflight'] = preflight['status']
    training[branch] = entry

evaluations = {}
for scope, folder in [('pilot', 'nr_push3_pilot_evaluations'), ('full', 'nr_push3_full_evaluations')]:
    for directory in sorted((ROOT / folder).glob('*')):
        if not directory.is_dir() or directory.name == 'preparation':
            continue
        entry = {'scope': scope}
        for filename in ['render_status.json', 'scores/status.json']:
            state = read(directory / filename)
            if state:
                entry[filename] = {k: state[k] for k in ['status', 'completed', 'expected', 'total'] if k in state}
        result = read(directory / 'scores/result.json')
        if result and result.get('status') == 'complete':
            entry.update(NR=result['test']['nr'], FR=result['validation']['fr'])
            entry['S_subset' if scope == 'pilot' else 'S_proxy'] = result['S_proxy']
        if len(entry) > 1:
            evaluations[directory.name] = entry

trained = all(v.get('stage') == 'complete' for v in training.values())
selection = read(ROOT / 'results/nr_push3_selection.json')
receipt = read(ROOT / 'results/nr_push3_backup_receipt_20261014.json')
phase = 'training' if not trained else 'evaluating'
if selection:
    phase = 'selected'
if receipt and receipt.get('status') == 'verified_complete':
    phase = 'complete'
snapshot = dict(status=phase,
    updated_at_bjt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    parent_full=dict(global_step=6500, optimizer_step=500, NR=2.2783103602882613,
                     FR=1.4614406824111938, S_proxy=3.739751042699455),
    scope='Three independent 750-update branches from the same6500 parent; pilot/full scores separate',
    training=training, evaluations=evaluations, selection=selection,
    backup=receipt['backup'] if receipt else None)
target = ROOT / 'results/status_cuda_nr_push3_20261014.json'
target.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(dict(status=phase, training={k: {f: v[f] for f in ['stage', 'completed_local_steps', 'global_step', 'optimizer_step', 'gradient_preflight'] if f in v} for k, v in training.items()},
                     evaluations=evaluations), ensure_ascii=False))
