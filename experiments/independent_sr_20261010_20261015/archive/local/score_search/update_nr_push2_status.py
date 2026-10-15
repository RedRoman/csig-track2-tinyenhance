"""Read-only progress snapshot of this bounded second NR optimization stage."""
from pathlib import Path
import datetime
import json

ROOT = Path('/cache/independent_sr_cuda_20261013')
training = {}
for name in ['cuda_nr_stage2_s6250_cont250', 'cuda_nr_stage2_s6250_roundste250',
             'cuda_nr_stage3_s6250_anchor6250_m1_250', 'cuda_nr_stage3_s6250_anchor6250_m15_250']:
    p = ROOT / 'adaptation/runs' / name / 'status.json'
    if p.exists():
        d = json.loads(p.read_text())
        training[name] = {k:d[k] for k in ['stage','completed_local_steps','total','global_step','optimizer_step'] if k in d}
        if 'row' in d:
            training[name].update(global_step=d['row']['global_step'], optimizer_step=d['row']['optimizer_step'],
                                  loss=d['row']['loss'], nonzero_lora_gradients=d['row']['nonzero_lora_gradients'])
    else:
        training[name] = {'stage':'initializing'}
evaluations = {}
for scope, folder in [('pilot','nr_push2_pilot_evaluations'),('full','nr_push2_full_evaluations')]:
    for d in sorted((ROOT / folder).glob('*')):
        if not d.is_dir() or d.name == 'preparation':
            continue
        entry = {'scope':scope}
        for f in ['render_status.json','scores/status.json']:
            p = d/f
            if p.exists():
                x = json.loads(p.read_text())
                entry[f] = {k:x[k] for k in ['status','completed','total','expected'] if k in x}
        p = d/'scores/result.json'
        if p.exists():
            x = json.loads(p.read_text())
            if x['status'] == 'complete':
                entry.update(test_NR=x['test']['nr'], val_FR=x['validation']['fr'])
                entry['S_subset' if scope=='pilot' else 'S_proxy'] = x['S_proxy']
        if len(entry)>1:
            evaluations[d.name] = entry
trained = len(training) == 4 and all(x.get('stage') == 'complete' for x in training.values())
quality_complete = (len(evaluations) == 11 and
                    all(('S_subset' if x['scope'] == 'pilot' else 'S_proxy') in x
                        for x in evaluations.values()))
backup_done = Path('/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014/DONE')
backup = json.loads(backup_done.read_text()) if backup_done.is_file() else {'status': 'not_started'}
stage = 'training' if not trained else 'evaluating'
if quality_complete:
    stage = 'quality_complete'
    if backup.get('status') == 'complete' and backup.get('finalize_pending') is False:
        stage = 'complete'
result = dict(status=stage, updated_at_bjt=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              scope='paired250 continuations, parent6250-anchor objective forks and coefficient160/185 probes; pilot/full scopes separate',
              previous_best_full=dict(step=6250,NR=2.2663850433592487,FR=1.463716487089793,S_proxy=3.7301015304490415),
              training=training,evaluations=evaluations, backup=backup)
selection = ROOT/'results/nr_push2_selection.json'
if selection.is_file():
    result['selection'] = json.loads(selection.read_text())
(ROOT/'results/status_cuda_nr_push2_20261014.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(training=training,evaluations=evaluations),ensure_ascii=False))
