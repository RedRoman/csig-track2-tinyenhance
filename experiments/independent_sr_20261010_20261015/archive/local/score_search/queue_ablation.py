"""Queue two declared ablations only after their original pilot dependencies.

No inference, tensor or NPU APIs are used here. A dependency must have both
complete result/status, exactly validation3+fixed20 (98 scalar measurements),
an exited controller PID, and a released device lock. The child controller
then acquires its own normal device lock and waits for actual idle hardware.
Only cards2/3 are used; this does not compete for cards0/1/5 or stop any job.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import time

PILOT = [1, 6, 7, 11, 16, 18, 21, 23, 25, 28, 31, 36, 41, 46, 51, 61, 71, 81, 91, 100]
SPECS = [
    dict(physical_npu=2, dependency='s10000_mt200_ct200_pilot', latent='mode', patch=512),
    dict(physical_npu=3, dependency='s8000_mt200_ct175_pilot', latent='sample', patch=768),
]


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def process_identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rindex(')') + 2:].split()
    return dict(state=fields[0], start_ticks=fields[19])


def running(pid, start_ticks):
    state = process_identity(pid)
    return state is not None and state['state'] not in ('Z', 'X') and state['start_ticks'] == start_ticks


def complete_pilot(root, label):
    directory = root / 'evaluations' / label
    status_path, result_path = directory / 'status.json', directory / 'scores/result.json'
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text())
    if status.get('stage') == 'failed':
        raise RuntimeError(f"{label} failed: {status.get('error')}")
    if status.get('stage') != 'complete' or not result_path.is_file():
        return False
    result = json.loads(result_path.read_text())
    if result.get('status') != 'complete' or result['test']['n'] != 20 or result['val']['n'] != 3:
        raise RuntimeError(f'{label} has invalid pilot result/counts')
    if result.get('test_cases') != PILOT or 'S_subset' not in result or 'S_proxy' in result:
        raise RuntimeError(f'{label} has invalid pilot case scope')
    rows = [json.loads(line) for line in (directory / 'scores/measurements.jsonl').read_text().splitlines() if line.strip()]
    expected = [('validation', case) for case in range(1, 4)] + [('test', case) for case in PILOT]
    if [(row['split'], row['case']) for row in rows] != expected or sum(len(row['metrics']) for row in rows) != 98:
        raise RuntimeError(f'{label} has incomplete/out-of-order pilot measurements')
    return True


def device_lock_released(root, physical_npu):
    with (root / f'device_locks/npu{physical_npu}.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        # This is a probe only; run_ablation.py obtains the actual lifetime lock.
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default='/cache/independent_sr_score_search_20261013')
    ap.add_argument('--poll-seconds', type=float, default=20)
    args = ap.parse_args()
    if args.poll_seconds <= 0:
        ap.error('--poll-seconds must be positive')
    root = args.root.resolve()
    with (root / 'ablation_queue.lock').open('a') as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / 'ablation_queue.json'
        prior = json.loads(state_path.read_text()) if state_path.exists() else None
        if prior and prior.get('specs') != SPECS:
            raise RuntimeError('Existing queue specification changed')
        source_jobs = {job['label']: job for job in json.loads((root / 'search_jobs.json').read_text())['jobs']}
        jobs = prior['jobs'] if prior else []
        if not jobs:
            for spec in SPECS:
                source = source_jobs[spec['dependency']]
                if source['physical_npu'] != spec['physical_npu']:
                    raise RuntimeError('Dependency/card mapping differs from declared plan')
                identity = process_identity(source['pid'])
                jobs.append(dict(**spec, label=f"s8000_mt200_ct200_{spec['latent']}_p{spec['patch']}_pilot",
                                 stage='waiting_dependency', dependency_pid=source['pid'],
                                 dependency_start_ticks=identity['start_ticks'] if identity else None,
                                 child_pid=None, child_start_ticks=None))
        processes = {}

        def publish():
            terminal = all(job['stage'] in ('complete', 'failed') for job in jobs)
            failed = any(job['stage'] == 'failed' for job in jobs)
            stage = ('finished_with_failure' if failed else 'complete') if terminal else 'running'
            atomic_json(state_path, dict(stage=stage, pid=os.getpid(), time=time.time(), specs=SPECS, jobs=jobs))

        publish()
        while not all(job['stage'] in ('complete', 'failed') for job in jobs):
            for job in jobs:
                if job['stage'] in ('complete', 'failed'):
                    continue
                try:
                    if job['child_pid'] is not None:
                        process = processes.get(job['label'])
                        if process is not None:
                            returncode = process.poll()
                            if returncode is None:
                                continue
                            job['returncode'] = returncode
                        elif running(job['child_pid'], job['child_start_ticks']):
                            continue
                        if complete_pilot(root, job['label']):
                            job['stage'] = 'complete'
                        else:
                            raise RuntimeError('Ablation controller exited without a complete pilot')
                        continue
                    if not complete_pilot(root, job['dependency']):
                        job['stage'] = 'waiting_dependency'
                        continue
                    if job['dependency_start_ticks'] is not None and running(job['dependency_pid'], job['dependency_start_ticks']):
                        job['stage'] = 'waiting_dependency_exit'
                        continue
                    if not device_lock_released(root, job['physical_npu']):
                        job['stage'] = 'waiting_device_lock_release'
                        continue
                    command = [str(root / 'env/bin/python'), '-u', str(root / 'code/score_search/run_ablation.py'),
                               '--root', str(root), '--step', '8000', '--coeff-t', '200', '--test-cases', 'pilot',
                               '--physical-npu', str(job['physical_npu']), '--latent', job['latent'], '--patch', str(job['patch'])]
                    with (root / f"logs/{job['label']}.log").open('a') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    identity = process_identity(child.pid)
                    job.update(stage='running', child_pid=child.pid, child_start_ticks=identity['start_ticks'] if identity else None,
                               command=command, launched_time=time.time())
                    processes[job['label']] = child
                    print(json.dumps(dict(event='launched', **job), ensure_ascii=False), flush=True)
                    publish()
                except Exception as error:
                    job.update(stage='failed', error=f'{type(error).__name__}: {error}')
                    print(json.dumps(dict(event='failed', **job), ensure_ascii=False), flush=True)
            publish()
            if not all(job['stage'] in ('complete', 'failed') for job in jobs):
                time.sleep(args.poll_seconds)
        raise SystemExit(1 if any(job['stage'] == 'failed' for job in jobs) else 0)


if __name__ == '__main__':
    main()
