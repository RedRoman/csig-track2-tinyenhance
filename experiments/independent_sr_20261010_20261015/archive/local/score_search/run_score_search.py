"""Isolated sequential rendering/scoring for one declared CSIG candidate."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import re
import subprocess
import time


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='/cache/independent_sr_score_search_20261013', type=Path)
    ap.add_argument('--step', type=int, choices=[6000, 8000, 10000], required=True)
    ap.add_argument('--coeff-t', type=int, required=True)
    ap.add_argument('--test-cases', choices=['pilot', 'all'], required=True)
    ap.add_argument('--physical-npu', type=int, choices=range(8), required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    label = f's{args.step}_mt200_ct{args.coeff_t}_{args.test_cases}'
    dest = root / 'evaluations' / label
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'logs').mkdir(exist_ok=True)
    checkpoint = root / f'adaptation/runs/ascend_balanced/step{args.step}.pth'
    code = root / 'code/score_search'
    python = root / 'env/bin/python'
    if not (dest / 'data').exists():
        (dest / 'data').symlink_to(root / 'scoring_assets/data', target_is_directory=True)
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.physical_npu),
               ASCEND_OPP_PATH=str(root / 'runtime/opp'),
               TORCH_HOME=str(root / 'scoring_assets/weights/torch'),
               HF_HOME=str(root / 'hf_cache'), HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH'] = ':'.join([str(root / 'code'), env.get('PYTHONPATH', ''),
        '/usr/local/Ascend/cann-8.5.2/python/site-packages',
        '/usr/local/Ascend/cann-8.5.2/opp/built-in/op_impl/ai_core/tbe'])
    env['LD_LIBRARY_PATH'] = ':'.join([str(root / 'runtime/hostlibs/lib64'),
        str(root / 'runtime/hostlibs/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64'),
        env.get('LD_LIBRARY_PATH', '')])

    def status(stage, **extra):
        value = dict(stage=stage, label=label, time=time.time(),
                     physical_npu=args.physical_npu, logical_npu=0,
                     checkpoint=str(checkpoint), scope=args.test_cases, **extra)
        atomic_json(dest / 'status.json', value)
        print(json.dumps(value), flush=True)

    def run(stage, command):
        status(stage)
        with (dest / f'logs/{stage}.log').open('a') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    locks = root / 'device_locks'
    locks.mkdir(exist_ok=True)
    with (dest / 'job.lock').open('a') as job_lock, (locks / f'npu{args.physical_npu}.lock').open('a') as device_lock:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                report = subprocess.check_output(['npu-smi', 'info'], text=True)
                idle = {int(x) for x in re.findall(r'No running processes found in NPU (\d+)', report)}
                if args.physical_npu in idle:
                    break
                status('waiting_for_idle_npu')
                time.sleep(30)
            if not (dest / 'render_complete').exists():
                run('rendering', [str(python), '-u', str(code / 'render_search.py'),
                    '--project-root', str(root), '--eval-root', str(dest),
                    '--checkpoint', str(checkpoint), '--test-cases', args.test_cases,
                    '--model-t', '200', '--coeff-t', str(args.coeff_t), '--backend', 'npu'])
            result = dest / 'scores/result.json'
            if not result.exists():
                run('scoring', [str(python), '-u', str(code / 'score_search.py'),
                    '--tool-dir', str(code), '--metric-code-dir', str(root / 'code'),
                    '--weights-root', str(root / 'scoring_assets/weights'), '--npu-index', '0',
                    '--test-dir', str(dest / 'render/test'), '--val-dir', str(dest / 'render/val'),
                    '--gt-dir', str(root / 'scoring_assets/data/验证集'), '--label', label,
                    '--out-dir', str(dest / 'scores'), '--test-cases', args.test_cases,
                    '--backend', 'npu'])
            value = json.loads(result.read_text())
            expected_n = 100 if args.test_cases == 'all' else 20
            if value.get('status') != 'complete' or value['test']['n'] != expected_n or value['val']['n'] != 3:
                raise RuntimeError('Scoring scope/count mismatch')
            score_key = 'S_proxy' if args.test_cases == 'all' else 'S_subset'
            status('complete', test_NR=value['test']['nr'], val_FR=value['val']['fr'],
                   **{score_key: value[score_key]}, result=str(result))
        except Exception as error:
            status('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
