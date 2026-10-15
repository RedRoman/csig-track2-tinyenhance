"""Actual CUDA HYPIR submission Runner; no CPU image return or NPU adaptation."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time


def _resolve(value, model_dir):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else model_dir / path).resolve()


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class Runner:
    """Runner(modeldir) reads modeldir/config.json; optional config for smoke CLI.

    infer input is a CUDA FP16 tensor [1,3,H,W] in [-1,1]. A uniform global downscale feeds the complete original small-image enhance
    pipeline, internally BF16. Native output is input plus the upsampled small
    enhancement residual (or explicit direct mode), CUDA FP32 [-1,1].
    This changes native pixels and random consumption. It remains stochastic:
    init seed231 then continuous upstream posterior.sample/DDPM noise calls.
    """

    def __init__(self, modeldir, config=None):
        self.model_dir = Path(modeldir).resolve()
        path = Path(config).resolve() if config else self.model_dir / 'config.json'
        self.config = json.loads(path.read_text())
        c = self.config
        if any(c.get(k, v) != v for k, v in {
            'seed': 231, 'lora_rank': 256, 'model_t': 200,
            'upscale': 1,
            'caption': '', 'internal_precision': 'bfloat16',
        }.items()):
            raise ValueError('This Runner requires the declared seed231/rank256/BF16/x1 protocol')
        self.compute_scale = float(c['compute_scale'])
        self.patch_size, self.stride = int(c['patch_size']), int(c['stride'])
        expected = {0.75: (384, 192), 0.625: (320, 160), 0.5: (256, 128)}
        if self.compute_scale not in expected or (self.patch_size, self.stride) != expected[self.compute_scale]:
            raise ValueError('Only declared uniform .75/p384s192, .625/p320s160 or .5/p256s128 compute policies are allowed')
        self.reconstruction_mode = c.get('reconstruction_mode', 'residual')
        if self.reconstruction_mode not in ('residual', 'direct'):
            raise ValueError('reconstruction_mode must be residual or direct')
        self.repo = _resolve(c['repo_path'], self.model_dir)
        self.base_model = _resolve(c['base_model_path'], self.model_dir)
        self.weight_path = _resolve(c['weight_path'], self.model_dir)
        if not self.weight_path.is_file():
            raise FileNotFoundError(self.weight_path)
        sys.path.insert(0, str(self.repo))
        import torch
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise RuntimeError('A native CUDA PyTorch environment is required')
        self.device = torch.device(f"cuda:{int(c.get('device_index', 0))}")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(4)
        from accelerate.utils import set_seed
        from HYPIR.enhancer.sd2 import SD2Enhancer
        import HYPIR.enhancer.base as base_module
        # Refuse accidentally imported StrSR/original CPU-return code.
        if Path(base_module.__file__).resolve() != self.repo / 'HYPIR/enhancer/base.py':
            raise RuntimeError('HYPIR module was already imported from another repository')
        source = Path(base_module.__file__).read_text()
        if 'return x.clamp(0, 1).cpu()' in source or 'return x.clamp(0, 1)' not in source:
            raise RuntimeError('CUDA repository must contain the recorded GPU-return-only patch')
        set_seed(231)  # Before construction and LoRA initialization, not after.
        modules = 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj'
        self.model = SD2Enhancer(base_model_path=str(self.base_model), weight_path=str(self.weight_path),
                                 lora_modules=modules.split(','), lora_rank=256,
                                 model_t=200, coeff_t=int(c.get('coeff_t', 175)), device=str(self.device))
        self.model.init_models()
        # Fixed empty-caption input preparation has no random operations.
        # Keep the exact original text encoder/timestep arithmetic, once.
        self.model.prepare_inputs(batch_size=1, prompt='')
        cached_inputs = self.model.inputs
        def prepare_cached_inputs(batch_size, prompt):
            if batch_size != 1 or prompt != '':
                raise ValueError('Cached Runner is restricted to one image and empty caption')
            self.model.inputs = cached_inputs
        self.model.prepare_inputs = prepare_cached_inputs
        self.fused = bool(c.get('fuse_lora', False))
        if self.fused:
            # BF16 merged weights change rounding/order: separate quality version.
            self.model.G.fuse_lora(lora_scale=1.0, safe_fusing=True, adapter_names=['default'])
            self.model.G.unload_lora()
        if not self.fused:
            raise ValueError('Compiled variant is the separate cache+fused quality version')
        base_module.wavelet_reconstruction = torch.compile(
            base_module.wavelet_reconstruction, mode='reduce-overhead', dynamic=False)
        # Compile only repeated fixed tile functions; the native image loop,
        # posterior sampling and scheduler step retain their original Python path.
        self.model.G.forward = torch.compile(self.model.G.forward, mode='reduce-overhead', dynamic=False)
        self.model.vae.encode = torch.compile(self.model.vae.encode, mode='reduce-overhead', dynamic=False)
        self.model.vae.decode = torch.compile(self.model.vae.decode, mode='reduce-overhead', dynamic=False)
        self._path = f"HYPIR_sd2_rank256_mt200_ct{self.model.coeff_t}_bf16_fused_gpu_cachedempty_ro_wavelet_scaled_s{self.compute_scale}_{self.reconstruction_mode}"
        versions = {}
        for name in ['torch', 'torchvision', 'diffusers', 'transformers', 'peft', 'accelerate', 'numpy', 'Pillow']:
            versions[name] = importlib.metadata.version(name)
        self.protocol = dict(
            schema='hypir_cuda_runner_v1', label=c.get('label'), runner_path=self._path,
            weight_path=str(self.weight_path), checkpoint_sha256=_sha(self.weight_path),
            base_model_path=str(self.base_model), repo_path=str(self.repo), versions=versions,
            input='CUDA FP16 [-1,1] native -> FP32[0,1] -> uniform bicubicAA downscale; no input/case recognition',
            output='CUDA FP32 [-1,1] native shape; final clamp[0,1] after uniform residual/direct reconstruction; no CPU image copy',
            internal_precision='BF16', seed='231 before initialization; continuous inference RNG',
            model_t=200, coeff_t=self.model.coeff_t, lora_rank=256, patch=self.patch_size, stride=self.stride,
            scale=1, caption='', latent='upstream posterior.sample',
            fusion='UNet fuse_lora(scale1,safe_fusing=True)+unload_lora' if self.fused else 'disabled',
            fused_requires_separate_quality_score=self.fused,
            source_sha256={str(p.relative_to(self.repo)): _sha(p) for p in [
                self.repo / 'HYPIR/enhancer/base.py', self.repo / 'HYPIR/enhancer/sd2.py',
                self.repo / 'HYPIR/utils/common.py']},
            runner_sha256=_sha(Path(__file__)),
            numerical_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                 cudnn_benchmark=torch.backends.cudnn.benchmark),
            no_warmup_during_initialization=True,
            fixed_empty_caption_cache='original prepare_inputs(batch1,prompt empty) once; identical tensors reused',
            compilation='UNet.forward,VAE.encode,VAE.decode and complete wavelet reduce-overhead dynamicFalse; original tile loops/sampling/scheduler',
            speed_candidate='push7 same A7750 weights; uniform global compute scale plus linked patch/stride; NO inherited quality score',
            compute_scale=self.compute_scale, reconstruction_mode=self.reconstruction_mode,
            small_size='max(8, 8*round(native_dim*compute_scale/8)); Python ties-to-even per axis, all inputs',
            resampling='FP32 bicubic align_cornersFalse antialiasTrue for downscale and native reconstruction',
            reconstruction='native_lq+upsample(enhanced_small-small_lq), or declared direct upsample(enhanced_small); only final clamp',
            changed_pixels_and_random_consumption=True, quality_reuse_allowed=False,
            compiled_requires_separate_quality_score=True,
        )

    def infer(self, x):
        import torch
        if not isinstance(x, torch.Tensor) or x.dtype != torch.float16 or x.device != self.device:
            raise TypeError('infer expects FP16 on the configured CUDA device')
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
            raise ValueError('infer expects [1,3,H,W]')
        torch.compiler.cudagraph_mark_step_begin()
        from torch.nn import functional as F
        with torch.no_grad():
            native_lq = (x.float() + 1.0) * 0.5
            h, w = x.shape[-2:]
            hs = max(8, 8 * round(h * self.compute_scale / 8))
            ws = max(8, 8 * round(w * self.compute_scale / 8))
            # Every input follows this identical formula; no benchmark/test shape branch.
            small_lq = F.interpolate(native_lq, size=(hs, ws), mode='bicubic',
                                     align_corners=False, antialias=True)
            small_output = self.model.enhance(lq=small_lq, prompt='', scale_by='factor', upscale=1,
                                             patch_size=self.patch_size, stride=self.stride, return_type='pt')
            if small_output.device != self.device or small_output.shape != small_lq.shape:
                raise RuntimeError('HYPIR returned incorrect small output device or shape')
            if self.reconstruction_mode == 'residual':
                small_delta = small_output - small_lq
                native_output = native_lq + F.interpolate(small_delta, size=(h, w), mode='bicubic',
                                                          align_corners=False, antialias=True)
            else:
                native_output = F.interpolate(small_output, size=(h, w), mode='bicubic',
                                              align_corners=False, antialias=True)
            if native_output.device != self.device or native_output.shape != x.shape:
                raise RuntimeError('Reconstruction returned incorrect native device or shape')
            return native_output.clamp(0, 1) * 2.0 - 1.0



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--smoke', action='store_true', help='Exactly one native CUDA512 forward, no benchmark/full render')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    import torch
    began = time.time()
    r = Runner(a.model_dir, a.config)
    torch.cuda.synchronize(r.device)
    report = dict(status='initialized', init_s=time.time()-began, protocol=r.protocol)
    if a.smoke:
        x = torch.zeros((1, 3, 512, 512), device=r.device, dtype=torch.float16)
        began = time.time();y = r.infer(x);torch.cuda.synchronize(r.device)
        finite = bool(torch.isfinite(y).all().item())
        if not finite:
            raise RuntimeError('Nonfinite smoke output')
        report.update(status='smoke_complete', forward_s=time.time()-began,
                      output=dict(device=str(y.device), dtype=str(y.dtype), shape=list(y.shape),
                                  finite=finite, min=float(y.min().item()), max=float(y.max().item())))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k != 'protocol'}), flush=True)


if __name__ == '__main__':
    main()
