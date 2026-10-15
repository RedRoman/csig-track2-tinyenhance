"""Launch the declared experiment only after migration validation succeeds."""
from pathlib import Path
import fcntl
import json
import subprocess
import time

ROOT = Path('/cache/independent_sr_score_search_20261013')
CODE = ROOT / 'code/score_search'


def main():
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    with (ROOT / 'search_launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (ROOT / 'migration/ready').exists():
            time.sleep(20)
        plan = json.loads((CODE / 'experiment_plan.json').read_text())
        records = []
        for job in plan['jobs']:
            label = f"s{job['step']}_mt200_ct{job['coeff_t']}_{job['test_cases']}"
            cmd = [str(ROOT / 'env/bin/python'), '-u', str(CODE / 'run_score_search.py'),
                   '--root', str(ROOT), '--step', str(job['step']),
                   '--coeff-t', str(job['coeff_t']), '--test-cases', job['test_cases'],
                   '--physical-npu', str(job['physical_npu'])]
            with (logs / f'{label}.log').open('a') as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            records.append(dict(label=label, pid=process.pid, command=cmd, **job))
        value = dict(stage='launched', time=time.time(), jobs=records)
        (ROOT / 'search_jobs.json').write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
