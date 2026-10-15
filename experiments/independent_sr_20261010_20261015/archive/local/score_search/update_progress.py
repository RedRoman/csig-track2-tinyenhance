"""Publish compact, scope-labelled progress without inferring unfinished scores."""
from pathlib import Path
import json
import time

ROOT = Path('/cache/independent_sr_score_search_20261013')


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    rows = []
    for directory in sorted((ROOT / 'evaluations').glob('*')):
        status = read_json(directory / 'status.json')
        render = read_json(directory / 'render_status.json')
        stream = read_json(directory / 'scores/stream_status.json')
        result = read_json(directory / 'scores/result.json')
        measurements = directory / 'scores/measurements.jsonl'
        n = len(measurements.read_text().splitlines()) if measurements.exists() else 0
        rows.append(dict(label=directory.name, stage=status.get('stage'),
                         render_completed=render.get('completed'), render_total=render.get('total'),
                         scored_images=n, stream_stage=stream.get('stage'),
                         complete=result.get('status') == 'complete',
                         test_n=result.get('test', {}).get('n'),
                         test_NR=result.get('test', {}).get('nr'),
                         val_FR=result.get('val', {}).get('fr'),
                         S_proxy=result.get('S_proxy'), S_subset=result.get('S_subset')))
    full = [row for row in rows if row['complete'] and row['test_n'] == 100]
    subset = [row for row in rows if row['complete'] and row['test_n'] == 20]
    value = dict(time=time.time(), root=str(ROOT), backend='Ascend NPU',
                 known_original_full_baseline=3.6603779575546587,
                 best_completed_full=max(full, key=lambda x: x['S_proxy']) if full else None,
                 best_completed_pilot=max(subset, key=lambda x: x['S_subset']) if subset else None,
                 predicted_leaderboard_score=None, cuda_latency_measured=False, candidates=rows)
    target = ROOT / 'results/status_score_search.json'
    target.parent.mkdir(exist_ok=True)
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(target)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
