"""Read-only CPU snapshot of Push4 runs and evaluations; writes a new status JSON."""
from pathlib import Path
import datetime
import json

ROOT = Path('/cache/independent_sr_cuda_20261013')
RUNS = {
    'A': 'cuda_nr_stage5_s7250_teacher6500_continue500',
    'B': 'cuda_nr_stage5_s7250_teacher6500_m25_500',
    'C': 'cuda_nr_stage5_s7250_teacher6500_m25_r10_500',
    'D': 'cuda_nr_stage5_s7250_teacher6500_m25_r075_250',
}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    training, evaluations = {}, {}
    for key, name in RUNS.items():
        run = ROOT / 'adaptation/runs' / name
        state = read(run / 'status.json') or {'stage': 'initializing'}
        record = {'run': str(run), 'status': state,
                  'gradient_preflight': read(run / 'gradient_preflight.json'),
                  'milestones': [read(run / f'checkpoint_step{step}.json') for step in (7500, 7750)
                                 if (run / f'checkpoint_step{step}.json').is_file()]}
        training[key] = record
    for scope in ('pilot', 'full'):
        for directory in sorted((ROOT / f'nr_push4_{scope}_evaluations').glob('push4_*')):
            if not directory.is_dir():continue
            evaluations[directory.name] = {
                'scope': scope, 'directory': str(directory),
                'render': read(directory / 'render_status.json'),
                'score': read(directory / 'scores/status.json'),
                'result': read(directory / 'scores/result.json'),
            }
    selection = read(ROOT / 'results/nr_push4_selection.json')
    receipt = read(ROOT / 'results/nr_push4_backup_receipt_20261014.json')
    phase = 'evaluating' if all(v['status'].get('stage') == 'complete' for v in training.values()) else 'training'
    if selection:phase = 'selected'
    if receipt and receipt.get('status') == 'verified_complete':phase = 'complete'
    snapshot = {'schema': 'cuda_nr_push4_live_status_v1', 'status': phase,
                'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'parent_global_step': 7250, 'parent_optimizer_step': 1250,
                'parent_full_NR': 2.313125129005856, 'parent_full_FR': 1.4465690056482952,
                'parent_full_S_proxy': 3.7596941346541515,
                'training': training, 'evaluations': evaluations,
                'selection': selection, 'backup': receipt.get('backup', receipt) if receipt else None}
    target = ROOT / 'results/status_cuda_nr_push4_20261014.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps({'phase': phase, 'training': {k: v['status'] for k, v in training.items()},
                      'evaluation_labels': list(evaluations)}, ensure_ascii=False))


if __name__ == '__main__':main()
