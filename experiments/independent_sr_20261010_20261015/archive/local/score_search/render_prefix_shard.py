"""Isolated HYPIR full-order shard renderer with exact RNG-call prefix replay.

No upstream files are changed. Validation must pass known full-order PNG byte
hashes before this method is used for a new candidate. Shard completion is not
full-render completion; this script never writes render_complete.
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', required=True, type=Path)
    ap.add_argument('--eval-root', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--model-t', type=int, default=200)
    ap.add_argument('--coeff-t', type=int, default=175)
    ap.add_argument('--task-start', type=int, default=0)
    ap.add_argument('--task-stop', type=int, default=103)
    ap.add_argument('--task-indices', help='Comma-separated full-order indices; validation only')
    ap.add_argument('--reference-render-root', type=Path)
    ap.add_argument('--backend', choices=['npu', 'cuda'], default='npu')
    ap.add_argument('--device-index', '--npu-index', type=int, default=0, dest='device_index')
    a = ap.parse_args()
    root, dest, checkpoint = (p.resolve() for p in (a.project_root, a.eval_root, a.checkpoint))
    repo = root / 'repos/HYPIR'
    sys.path.insert(0, str(repo))
    import torch
    if a.backend == 'npu':
        import torch_npu
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    accelerator = getattr(torch, a.backend)
    accelerator.set_device(a.device_index)
    import numpy as np
    from PIL import Image
    from accelerate.utils import set_seed
    from torchvision.transforms import ToTensor
    from HYPIR.enhancer.sd2 import SD2Enhancer
    from HYPIR.utils.common import sliding_windows
    import diffusers.models.autoencoders.vae as vae_module
    import diffusers.schedulers.scheduling_ddpm as ddpm_module
    import diffusers.utils.torch_utils as random_module
    Image.MAX_IMAGE_PIXELS = None
    torch.set_num_threads(4)
    tasks = [('val', i, root / f'scoring_assets/data/验证集/case{i}_lq.jpg') for i in range(1, 4)]
    tasks += [('test', i, root / f'assets/input/case{i}.jpg') for i in range(1, 101)]
    selected = ([int(x) for x in a.task_indices.split(',')] if a.task_indices
                else list(range(a.task_start, a.task_stop)))
    if not selected or selected != sorted(set(selected)) or min(selected) < 0 or max(selected) >= 103:
        raise ValueError('Task indices must be unique, increasing full-order indices in [0,103)')
    if a.task_indices and not a.reference_render_root:
        raise ValueError('Noncontiguous indices require a known reference for validation')
    task_manifest = []
    for split, case, path in tasks:
        with Image.open(path) as image:
            size = list(image.size)
        task_manifest.append(dict(split=split, case=case, input_sha256=digest(path), size=size))
    sources = [Path(__file__).resolve(), repo / 'HYPIR/enhancer/sd2.py',
               repo / 'HYPIR/enhancer/base.py', repo / 'HYPIR/utils/common.py',
               Path(vae_module.__file__), Path(ddpm_module.__file__), Path(random_module.__file__)]
    protocol = dict(schema='hypir_full_rng_prefix_shard_v1', checkpoint_sha256=digest(checkpoint),
                    tasks=task_manifest, test_cases=list(range(1, 101)), model_t=a.model_t,
                    coeff_t=a.coeff_t, latent='sample', patch=512, stride=256,
                    precision='upstream internal BF16', scale=1, caption='',
                    inference_seed='231 before model initialization; continuous full103 order',
                    prefix='actual VAE randn_tensor then actual DDPM randn_tensor per real tile; generator=None',
                    output='upstream PIL uint8-floor, lossless PNG compress_level1, native size',
                    backend=a.backend, logical_device=a.device_index, torch=torch.__version__,
                    base_model=str(root / 'models/Manojb__stable-diffusion-2-1-base'),
                    source_sha256={str(p): digest(p) for p in sources})
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'shard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = dest / 'shard_progress.pt'
        saved = torch.load(progress, map_location='cpu', weights_only=False) if progress.exists() else None
        records = [] if saved is None else saved['records']
        if saved is not None:
            if saved['protocol_id'] != identity or saved['selected_indices'] != selected:
                raise RuntimeError('Shard protocol or selection changed')
            for record, index in zip(records, selected):
                if record['task_index'] != index or digest(dest / record['output']) != record['sha256']:
                    raise RuntimeError('Committed shard output changed')
        if len(records) > len(selected):
            raise RuntimeError('Invalid shard cursor')
        if len(records) < len(selected):
            set_seed(231)
            modules = 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj'
            model = SD2Enhancer(base_model_path=protocol['base_model'], weight_path=str(checkpoint),
                               lora_modules=modules.split(','), lora_rank=256,
                               model_t=a.model_t, coeff_t=a.coeff_t, device='cuda')
            model.init_models()
            if model.weight_dtype != torch.bfloat16 or model.vae.config.latent_channels != 4:
                raise RuntimeError('Replay requires audited BF16/4-channel HYPIR')
            if getattr(model.vae, 'use_tiling', False) or getattr(model.vae, 'use_slicing', False):
                raise RuntimeError('Unexpected internal VAE tiling/slicing')
            if model.scheduler.config.variance_type in ['learned', 'learned_range']:
                raise RuntimeError('Replay only validated for four-channel fixed-variance scheduler')
            device = next(model.vae.parameters()).device
            if saved:
                random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
                torch.set_rng_state(saved['torch_rng'])
                accelerator.set_rng_state(saved['device_rng'], a.device_index)
                cursor = records[-1]['task_index'] + 1
            else:
                cursor = 0

            def replay(index):
                # Mirrors BaseEnhancer.scale1, resize_at_least and padding.
                w, h = task_manifest[index]['size']
                if min(h, w) <= 512:
                    if h == w:
                        h, w = 512, 512
                    elif h < w:
                        h, w = 512, int(w * (512 / h))
                    else:
                        h, w = int(h * (512 / w)), 512
                h, w = (h + 7) // 8 * 8, (w + 7) // 8 * 8
                enc = sliding_windows(h, w, 512, 256)
                gen = sliding_windows(h // 8, w // 8, 64, 32)
                # All encode tiles precede all generator tiles (not interleaved).
                for hi, he, wi, we in enc:
                    noise = vae_module.randn_tensor((1, 4, (he-hi)//8, (we-wi)//8),
                                                    generator=None, device=device, dtype=torch.bfloat16)
                    del noise
                if a.coeff_t > 0:
                    for hi, he, wi, we in gen:
                        noise = ddpm_module.randn_tensor((1, 4, he-hi, we-wi),
                                                         generator=None, device=device, dtype=torch.bfloat16)
                        del noise
                return dict(task_index=index, encode_draws=len(enc), scheduler_draws=len(gen) if a.coeff_t > 0 else 0)

            to_tensor = ToTensor()
            for index in selected[len(records):]:
                begin = time.time(); replayed = []
                while cursor < index:
                    replayed.append(replay(cursor)); cursor += 1
                accelerator.synchronize()
                print(json.dumps(dict(stage='prefix_replayed', next_index=index,
                                      images=len(replayed), draws=sum(x['encode_draws']+x['scheduler_draws'] for x in replayed),
                                      seconds=time.time()-begin)), flush=True)
                split, case, path = tasks[index]
                with Image.open(path) as image:
                    image = image.convert('RGB'); expected = image.size
                    tensor = to_tensor(image).unsqueeze(0)
                if list(expected) != task_manifest[index]['size'] or digest(path) != task_manifest[index]['input_sha256']:
                    raise RuntimeError('Input changed')
                result = model.enhance(lq=tensor, prompt='', scale_by='factor', upscale=1,
                                       patch_size=512, stride=256, return_type='pil')[0]
                if result.size != expected:
                    raise RuntimeError('Output size changed')
                output = dest / f'render/{split}/case{case}.png';output.parent.mkdir(parents=True, exist_ok=True)
                temp = output.with_suffix('.tmp');result.save(temp, format='PNG', compress_level=1)
                os.replace(temp, output)
                row = dict(task_index=index, split=split, case=case, output=str(output.relative_to(dest)),
                           sha256=digest(output), size=list(expected), seconds=time.time()-begin)
                if a.reference_render_root:
                    reference = a.reference_render_root / row['output']
                    row['reference_sha256'] = digest(reference)
                    row['reference_byte_exact'] = row['sha256'] == row['reference_sha256']
                    if not row['reference_byte_exact']:
                        atomic_json(dest / 'shard_status.json', dict(stage='validation_failed', record=row))
                        raise RuntimeError('RNG-prefix output differs from serial reference PNG')
                records.append(row);cursor=index+1
                state = dict(protocol_id=identity, selected_indices=selected, records=records,
                             python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                             torch_rng=torch.get_rng_state(), device_rng=accelerator.get_rng_state(a.device_index))
                temporary = progress.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    torch.save(state, handle);handle.flush();os.fsync(handle.fileno())
                os.replace(temporary, progress)
                atomic_json(dest / 'shard_status.json', dict(stage='rendering', completed=len(records), total=len(selected), latest=row))
                print(json.dumps(row), flush=True)
                del tensor, result;accelerator.empty_cache()
        manifest = dict(protocol=protocol, protocol_id=identity, checkpoint=str(checkpoint),
                        checkpoint_sha256=protocol['checkpoint_sha256'], selected_indices=selected,
                        records=records, full_order_count=103, validation=bool(a.reference_render_root))
        atomic_json(dest / 'shard_manifest.json', manifest)
        atomic_json(dest / 'shard_status.json', dict(stage='validation_passed' if a.reference_render_root else 'shard_complete',
                                                  completed=len(records), total=len(selected), full_render_complete=False))
        (dest / 'shard_complete').touch()
        print('SHARD_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
