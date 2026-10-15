"""Read-only job progress, published separately from formal completed scores."""
from pathlib import Path
import datetime
import json

R = Path('/cache/independent_sr_cuda_20261013')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    evaluations = {}
    for root in [R/'nr_push5_full_evaluations', R/'nr_push5_pilot_evaluations']:
        if not root.exists():
            continue
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not directory.name.startswith('push5_'):
                continue
            render = read(directory/'render_status.json')
            result = read(directory/'scores/result.json')
            record = {'render':render,'scored_images':result.get('completed',0) if result else 0,
                      'complete':bool(result and result.get('status')=='complete')}
            if record['complete']:
                record.update(NR=result['test']['nr'], FR=result['validation']['fr'])
                record['S_proxy' if result.get('official_full_count') else 'S_subset'] = result['S_proxy']
            evaluations[directory.name] = record
    runs = {}
    for run in sorted((R/'adaptation/runs').glob('*push5*')):
        if run.is_dir():
            runs[run.name] = read(run/'status.json')
    selection = read(R/'results/nr_push5_selection.json')
    receipt = read(R/'results/nr_push5_backup_receipt_20261014.json')
    stage = 'complete' if receipt and receipt.get('status')=='verified_complete' else 'selected' if selection else 'running'
    status = {'schema':'cuda_nr_push5_status_v1','status':stage,'target_original_tool_prediction':70,
              'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'strategy':'HYPIR speed experiment plus independent supplied-G4-initialized pixel NR microtuning',
              'reference_is_new_training':False,'evaluations':evaluations,'training_runs':runs,
              'hypir_speed_experiment':read(R/'calibration/push5_hypir_clro_wavelet_latency.json'),
              'selection':selection,'backup':receipt}
    path = R/'results/status_cuda_nr_push5_20261014.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)
    counts = {k:{'rendered':v['render'].get('completed',0) if v['render'] else 0,
                 'scored':v['scored_images'],'complete':v['complete']} for k,v in evaluations.items()}
    print(json.dumps({'status':stage,'evaluations':counts},ensure_ascii=False))


if __name__=='__main__':
    main()
