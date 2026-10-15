"""Render isolated HYPIR candidates; do not change training or existing results.

pilot = validation1..3 + fixed20 numeric test cases; all = validation3+test100.
Seed231 is set BEFORE HYPIR construction/initialization, matching the existing
8000 renderer. Outputs retain upstream PIL/uint8-floor encoding, native size,
BF16 internal precision, x1 and empty prompt. No latency warmup is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import sys
import time
import fcntl

PILOT_CASES = [1, 6, 7, 11, 16, 18, 21, 23, 25, 28, 31, 36, 41, 46, 51, 61, 71, 81, 91, 100]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', required=True, type=Path)
    ap.add_argument('--eval-root', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--test-cases', choices=['pilot', 'all'], default='pilot')
    ap.add_argument('--model-t', type=int, default=200)
    ap.add_argument('--coeff-t', type=int, default=200)
    ap.add_argument('--backend', choices=['auto', 'cuda', 'npu'], default='auto')
    ap.add_argument('--device-index', '--npu-index', type=int, default=0, dest='device_index')
    a = ap.parse_args()
    root, dest, checkpoint = (p.resolve() for p in (a.project_root, a.eval_root, a.checkpoint))
    repo = root / 'repos/HYPIR'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(repo))
    import torch
    backend = a.backend if a.backend != 'auto' else ('cuda' if torch.cuda.is_available() else 'npu')
    if backend == 'npu':
        import torch_npu
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    accelerator = getattr(torch, backend)
    accelerator.set_device(a.device_index)
    import numpy as np
    from PIL import Image
    from accelerate.utils import set_seed
    from torchvision.transforms import ToTensor
    from HYPIR.enhancer.sd2 import SD2Enhancer
    Image.MAX_IMAGE_PIXELS = None
    torch.set_num_threads(4)
    cases = PILOT_CASES if a.test_cases == 'pilot' else list(range(1, 101))
    tasks = [('val', i, dest / f'data/验证集/case{i}_lq.jpg') for i in range(1, 4)]
    tasks += [('test', i, root / f'assets/input/case{i}.jpg') for i in cases]
    task_manifest = []
    for split, case, path in tasks:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            size = list(image.size)
        task_manifest.append(dict(split=split, case=case, input_sha256=digest(path), size=size))
    weight_hash = digest(checkpoint)
    inputs = [task['input_sha256'] for task in task_manifest]
    sources = [Path(__file__).resolve(), repo / 'HYPIR/enhancer/sd2.py',
               repo / 'HYPIR/enhancer/base.py', repo / 'HYPIR/utils/common.py']
    protocol = dict(checkpoint_sha256=weight_hash, tasks=task_manifest, test_cases=cases,
                    model_t=a.model_t, coeff_t=a.coeff_t, patch=512, stride=256,
                    precision='upstream internal BF16', scale=1, caption='',
                    inference_seed='231 before model initialization; continuous ordered images',
                    output='upstream PIL uint8-floor, lossless PNG, native size',
                    backend=backend, logical_device=a.device_index, torch=torch.__version__,
                    base_model=str(root / 'models/Manojb__stable-diffusion-2-1-base'),
                    source_sha256={str(p.relative_to(repo)) if p.is_relative_to(repo) else 'render_search.py': digest(p) for p in sources})
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'render.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress_file = dest / 'render_progress.pt'
        records, saved = [], None
        if progress_file.exists():
            saved = torch.load(progress_file, map_location='cpu', weights_only=False)
            if saved.get('protocol_id') != identity:
                raise RuntimeError('Checkpoint/input/inference/source identity changed; use a new evaluation directory')
            records = saved['records']
            if len(records) > len(tasks):
                raise RuntimeError('Invalid render cursor')
            for row, task in zip(records, task_manifest):
                output = dest / row['output']
                if (row['split'], row['case'], row['size']) != (task['split'], task['case'], task['size']):
                    raise RuntimeError('Committed render task changed')
                if not output.is_file() or digest(output) != row['sha256']:
                    raise RuntimeError(f'Committed output changed: {output}')
        elif (dest / 'render_complete').exists():
            raise RuntimeError('Completion marker without authoritative render state')
        if len(records) < len(tasks):
            set_seed(231)
            modules = 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj'
            model = SD2Enhancer(base_model_path=protocol['base_model'], weight_path=str(checkpoint),
                               lora_modules=modules.split(','), lora_rank=256,
                               model_t=a.model_t, coeff_t=a.coeff_t, device='cuda')
            model.init_models()
            if saved:
                random.setstate(saved['python_rng'])
                np.random.set_state(saved['numpy_rng'])
                torch.set_rng_state(saved['torch_rng'])
                accelerator.set_rng_state(saved['device_rng'], a.device_index)
            to_tensor = ToTensor()
            for index in range(len(records), len(tasks)):
                split, case, path = tasks[index]
                begin = time.time()
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    expected_size = image.size
                    tensor = to_tensor(image).unsqueeze(0)
                if list(expected_size) != task_manifest[index]['size'] or digest(path) != inputs[index]:
                    raise RuntimeError('Input changed during rendering')
                output = dest / f'render/{split}/case{case}.png'
                output.parent.mkdir(parents=True, exist_ok=True)
                result = model.enhance(lq=tensor, prompt='', scale_by='factor', upscale=1,
                                       patch_size=512, stride=256, return_type='pil')[0]
                if result.size != expected_size:
                    raise RuntimeError(f'Output size changed: {result.size} != {expected_size}')
                temp = output.with_suffix('.tmp')
                result.save(temp, format='PNG', compress_level=1)
                os.replace(temp, output)
                row = dict(split=split, case=case, output=str(output.relative_to(dest)),
                           sha256=digest(output), size=list(expected_size), seconds=time.time() - begin)
                records.append(row)
                state = dict(protocol_id=identity, checkpoint_sha256=weight_hash, input_sha256=inputs,
                             records=records, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                             torch_rng=torch.get_rng_state(), device_rng=accelerator.get_rng_state(a.device_index))
                temp_state = progress_file.with_suffix('.tmp')
                with temp_state.open('wb') as handle:
                    torch.save(state, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_state, progress_file)
                status = dict(stage='rendering', completed=len(records), total=len(tasks), latest=row,
                              time=time.time(), backend=backend, model_t=a.model_t, coeff_t=a.coeff_t,
                              checkpoint_sha256=weight_hash, test_cases=cases)
                (dest / 'render_status.json').write_text(json.dumps(status, indent=2))
                print(json.dumps(status), flush=True)
                del tensor, result
                accelerator.empty_cache()
        manifest = dict(checkpoint=str(checkpoint), protocol_id=identity, protocol=protocol,
                        checkpoint_sha256=weight_hash, input_sha256=inputs, records=records,
                        scope='test100+validation3' if a.test_cases == 'all' else 'fixed20+validation3 pilot; not full test100')
        (dest / 'render_manifest.json').write_text(json.dumps(manifest, indent=2))
        (dest / 'render_complete').touch()
        print('RENDER_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
