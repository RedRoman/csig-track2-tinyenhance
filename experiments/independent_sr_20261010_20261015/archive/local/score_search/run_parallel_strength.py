"""Render the selected strength candidate in audited full-order shards."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time

ROOT = Path('/cache/independent_sr_score_search_20261013')
DEST = ROOT / 'evaluations/s6000_mt200_ct175_all'
CODE = ROOT / 'code/score_search'
RANGES = [(0, 26), (26, 52), (52, 78), (78, 103)]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def environment(device):
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(device),
               ASCEND_OPP_PATH=str(ROOT / 'runtime/opp'),
               TORCH_HOME=str(ROOT / 'scoring_assets/weights/torch'),
               HF_HOME=str(ROOT / 'hf_cache'), HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH'] = ':'.join([str(ROOT / 'code'), env.get('PYTHONPATH', ''),
        '/usr/local/Ascend/cann-8.5.2/python/site-packages',
        '/usr/local/Ascend/cann-8.5.2/opp/built-in/op_impl/ai_core/tbe'])
    env['LD_LIBRARY_PATH'] = ':'.join([str(ROOT / 'runtime/hostlibs/lib64'),
        str(ROOT / 'runtime/hostlibs/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64'),
        env.get('LD_LIBRARY_PATH', '')])
    return env


def main():
    import torch
    proof_path = ROOT / 'evaluations/prefix_validation_s8000_case37_case99/shard_manifest.json'
    proof = json.loads(proof_path.read_text())
    assert proof['selected_indices'] == [39, 101]
    assert len(proof['records']) == 2 and all(x['reference_byte_exact'] for x in proof['records'])
    renderer = CODE / 'render_prefix_shard.py'
    assert proof['protocol']['source_sha256'][str(renderer)] == digest(renderer)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'logs').mkdir(exist_ok=True)
    with (DEST / 'parallel.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = ROOT / 'adaptation/runs/ascend_balanced/step6000.pth'
        checkpoint_hash = digest(checkpoint)
        tasks = [('val', i, ROOT / f'scoring_assets/data/验证集/case{i}_lq.jpg') for i in range(1, 4)]
        tasks += [('test', i, ROOT / f'assets/input/case{i}.jpg') for i in range(1, 101)]
        input_hashes = [digest(x[2]) for x in tasks]
        report = subprocess.check_output(['npu-smi', 'info'], text=True)
        idle = {int(x) for x in re.findall(r'No running processes found in NPU (\d+)', report)}
        assert {2, 3, 4, 5} <= idle, 'Required shard devices are occupied'
        jobs, handles, processes = [], [], []
        for shard, ((start, stop), device) in enumerate(zip(RANGES, [2, 3, 4, 5])):
            device_lock = (ROOT / f'device_locks/npu{device}.lock').open('a')
            fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(device_lock)
            directory = DEST / f'shards/shard{shard}'
            cmd = [str(ROOT / 'env/bin/python'), '-u', str(renderer),
                   '--project-root', str(ROOT), '--eval-root', str(directory),
                   '--checkpoint', str(checkpoint), '--model-t', '200', '--coeff-t', '175',
                   '--task-start', str(start), '--task-stop', str(stop),
                   '--backend', 'npu', '--device-index', '0']
            with (DEST / f'logs/shard{shard}.log').open('a') as log:
                p = subprocess.Popen(cmd, env=environment(device), stdout=log,
                                     stderr=subprocess.STDOUT)
            processes.append(p)
            jobs.append(dict(shard=shard, physical_npu=device, start=start, stop=stop,
                             pid=p.pid, command=cmd))
        write_json(DEST / 'parallel_jobs.json', dict(proof_sha256=digest(proof_path), jobs=jobs))
        committed, protocols = [], {}
        while len(committed) < 103:
            for index, p in enumerate(processes):
                if p.poll() is not None and p.returncode != 0:
                    raise RuntimeError(f'Shard {index} exited {p.returncode}; retain partial artifacts')
            active = next(i for i, (a, b) in enumerate(RANGES) if a <= len(committed) < b)
            directory = DEST / f'shards/shard{active}'
            progress = directory / 'shard_progress.pt'
            if not progress.exists():
                time.sleep(2)
                continue
            state = torch.load(progress, map_location='cpu', weights_only=False)
            begin, end = RANGES[active]
            assert state['selected_indices'] == list(range(begin, end))
            additions = state['records'][len(committed) - begin:]
            if not additions:
                time.sleep(2)
                continue
            for row in additions:
                assert row['task_index'] == len(committed)
                source, target = directory / row['output'], DEST / row['output']
                assert digest(source) == row['sha256']
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert digest(target) == row['sha256']
                else:
                    os.link(source, target)
                committed.append(row)
            # Each batch ends at the shard state cursor, so its post-image RNG
            # is the full-order RNG after the global last committed image.
            merged = dict(checkpoint_sha256=checkpoint_hash,
                          input_sha256=input_hashes, records=committed,
                          prefix_validation_sha256=digest(proof_path),
                          python_rng=state['python_rng'], numpy_rng=state['numpy_rng'],
                          torch_rng=state['torch_rng'], device_rng=state['device_rng'])
            tmp = DEST / 'render_progress.tmp'
            torch.save(merged, tmp)
            tmp.replace(DEST / 'render_progress.pt')
            write_json(DEST / 'render_status.json', dict(stage='rendering', completed=len(committed),
                       total=103, latest=committed[-1], time=time.time(), parallel_shards=4))
            print('COMMITTED', len(committed), flush=True)
        for i, p in enumerate(processes):
            p.wait()
            assert p.returncode == 0
            manifest = json.loads((DEST / f'shards/shard{i}/shard_manifest.json').read_text())
            protocols[i] = manifest['protocol']
        assert all(x == protocols[0] for x in protocols.values())
        assert digest(checkpoint) == checkpoint_hash
        assert [digest(x[2]) for x in tasks] == input_hashes
        for row in committed:
            assert digest(DEST / row['output']) == row['sha256']
        write_json(DEST / 'render_manifest.json', dict(checkpoint=str(checkpoint),
                   checkpoint_sha256=protocols[0]['checkpoint_sha256'],
                   input_sha256=[x['input_sha256'] for x in protocols[0]['tasks']],
                   protocol=protocols[0], records=committed,
                   parallel_method='four full-order contiguous shards; exact RNG prefix replay',
                   proof_sha256=digest(proof_path), controller_sha256=digest(Path(__file__))))
        (DEST / 'render_complete').touch()
        for h in handles:
            h.close()
        result = DEST / 'scores/result.json'
        while not result.exists():
            time.sleep(5)
        score = json.loads(result.read_text())
        assert score['status'] == 'complete' and score['test']['n'] == 100 and score['val']['n'] == 3
        write_json(DEST / 'status.json', dict(stage='complete', label=DEST.name,
                   test_NR=score['test']['nr'], val_FR=score['val']['fr'], S_proxy=score['S_proxy'], time=time.time()))


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    write_json(DEST / 'status.json', dict(stage='rendering_and_scoring', label=DEST.name,
               scope='all', time=time.time()))
    try:
        main()
    except Exception as error:
        write_json(DEST / 'status.json', dict(stage='failed', label=DEST.name, error=repr(error), time=time.time()))
        raise
