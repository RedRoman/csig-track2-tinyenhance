"""Fresh-process CUDA submission quality render; no latency warmup.

Matches supplied eval_submission.render input FP16 [-1,1], native size,
GPU Runner.infer, output clamp[-1,1] and rounded uint8 lossless PNG.
Seed231 is owned by the Runner before model initialization. Continuous RNG
is committed with every image so an interrupted run can resume exactly.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

PILOT = [1, 6, 7, 11, 16, 18, 21, 23, 25, 28, 31, 36, 41, 46, 51, 61, 71, 81, 91, 100]
VAL = ['case1_lq.jpg', 'case2_lq.jpg', 'case3_lq.jpg']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--test-dir', type=Path, required=True)
    ap.add_argument('--val-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--cases', choices=['all', 'pilot'], default='all')
    a = ap.parse_args()
    for k in ['model_dir', 'test_dir', 'val_dir', 'out_dir']:
        setattr(a, k, getattr(a, k).resolve())
    import torch
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    if not torch.cuda.is_available():
        raise RuntimeError('Native CUDA is required')
    torch.cuda.set_device(0)
    tasks = [('val', i, a.val_dir / name) for i, name in enumerate(VAL, 1)]
    tasks += [('test', i, a.test_dir / f'case{i}.jpg') for i in (range(1, 101) if a.cases == 'all' else PILOT)]
    inputs = []
    for split, case, p in tasks:
        with Image.open(p) as im:
            size = list(im.size)
        inputs.append(dict(split=split, case=case, input=str(p), input_sha256=sha(p), size=size))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / 'render.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = os.getcwd()
        sys.path.append(str(a.model_dir))
        os.chdir(a.model_dir)
        try:
            spec = importlib.util.spec_from_file_location('submission_runner', a.model_dir / 'runner.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            begin = time.time()
            runner = mod.Runner(str(a.model_dir))
            torch.cuda.synchronize()
            init_s = time.time() - begin
        finally:
            os.chdir(old)
        protocol = dict(schema='cuda_submission_quality_v1', runner=runner.protocol,
                        model_dir=str(a.model_dir), tasks=inputs, cases=a.cases,
                        input_encoding='original tool load_img float32[0,1] -> half()*2-1; native size',
                        output_encoding='original tool float().clamp(-1,1); ((*0.5+0.5)*255).round().astype(uint8); PNG compress1',
                        warmup=False, inference_rng='seed231 before model initialization; continuous ordered val3 then tests',
                        source_sha256=sha(Path(__file__)),
                        model_dir_sha256={str(p.relative_to(a.model_dir)): sha(p) for p in sorted(a.model_dir.rglob('*')) if p.is_file()},
                        gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        state_path = a.out_dir / 'render_progress.pt'
        records = []
        if state_path.exists():
            saved = torch.load(state_path, map_location='cpu', weights_only=False)
            if saved['protocol_id'] != identity:
                raise RuntimeError('Render identity changed; use a new output directory')
            records = saved['records']
            for row, task in zip(records, inputs):
                if (row['split'], row['case'], row['size']) != (task['split'], task['case'], task['size']):
                    raise RuntimeError('Committed image order changed')
                if sha(a.out_dir / row['output']) != row['sha256']:
                    raise RuntimeError('Committed PNG changed')
            random.setstate(saved['python_rng'])
            np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state(saved['device_rng'], 0)
        atomic_json(a.out_dir / 'render_protocol.json', dict(protocol_id=identity, protocol=protocol, init_s=init_s))
        for index in range(len(records), len(tasks)):
            split, case, src = tasks[index]
            begin = time.time()
            with Image.open(src) as im:
                array = np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0
            if sha(src) != inputs[index]['input_sha256']:
                raise RuntimeError('Source changed')
            x = torch.from_numpy(array).permute(2, 0, 1)[None].to('cuda').half() * 2 - 1
            with torch.no_grad():
                y = runner.infer(x).float().clamp(-1, 1)
            if y.shape != x.shape or not bool(torch.isfinite(y).all()):
                raise RuntimeError('Invalid native output')
            img = Image.fromarray(((y[0].permute(1, 2, 0).cpu().numpy() * .5 + .5) * 255).round().astype(np.uint8))
            dst = a.out_dir / f'render/{split}/case{case}.png'
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix('.tmp')
            img.save(tmp, format='PNG', compress_level=1)
            os.replace(tmp, dst)
            row = dict(task_index=index, split=split, case=case, output=str(dst.relative_to(a.out_dir)),
                       sha256=sha(dst), size=inputs[index]['size'], seconds=time.time() - begin)
            records.append(row)
            state = dict(protocol_id=identity, checkpoint_sha256=runner.protocol['checkpoint_sha256'],
                         input_sha256=[t['input_sha256'] for t in inputs], records=records,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), device_rng=torch.cuda.get_rng_state(0))
            tmpstate = state_path.with_suffix('.tmp')
            with tmpstate.open('wb') as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmpstate, state_path)
            status = dict(status='rendering', completed=len(records), total=len(tasks), latest=row,
                          time=time.time(), protocol_id=identity)
            atomic_json(a.out_dir / 'render_status.json', status)
            print(json.dumps(status), flush=True)
            del x, y, img, array
            torch.cuda.empty_cache()
        if len(records) != len(tasks):
            raise RuntimeError('Incomplete render')
        for row in records:
            if sha(a.out_dir / row['output']) != row['sha256']:
                raise RuntimeError('Final output hash mismatch')
        atomic_json(a.out_dir / 'render_manifest.json', dict(protocol_id=identity, protocol=protocol, records=records,
                                                          scope='test100+validation3' if a.cases == 'all' else 'fixed20+validation3 pilot'))
        atomic_json(a.out_dir / 'render_status.json', dict(status='complete', completed=len(records), total=len(tasks), time=time.time()))
        (a.out_dir / 'render_complete').touch()
        print('RENDER_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
