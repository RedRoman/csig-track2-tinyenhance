CUDA Runner uses the plain HYPIR source copied from `source_payload/repos/HYPIR`
to `/cache/independent_sr_cuda_20261013/repos/HYPIR_cuda`. The only upstream
source difference is `BaseEnhancer.enhance(return_type='pt')` returning
`x.clamp(0,1)` on GPU instead of copying it to CPU. The actual diff is saved at
`code/hypir_cuda_runner/gpu_return_only.diff` on the CUDA host.

The default candidate is step6000, model_t200, coeff_t175, rank256, BF16,
posterior.sample, patch512/stride256, empty caption, upscale1. The baseline
configuration is step8000/model_t200/coeff_t200. `Runner(modeldir)` reads
`modeldir/config.json`; the standalone smoke CLI also accepts `--config`.

Input and output follow the submission interface: input CUDA FP16 `[-1,1]`,
converted on GPU to FP32 `[0,1]`; output CUDA FP32 `[-1,1]` at native size.
This FP16 input conversion differs from the previous NPU quality renderer's
FP32 `[0,1]` input. Save fresh CUDA quality results under an independent label
and document the PNG rounding policy. CUDA/NPU score equivalence is not assumed.

Initialization sets seed231 before HYPIR construction and LoRA initialization.
There is no initialization warmup, cached text embedding, fused attention,
LoRA fusion, changed precision, image resize, or CPU image roundtrip. The
upstream text/encoder/generator/decoder/wavelet pipeline executes on every infer.
Quality rendering must construct a fresh Runner in a separate process from
latency warm30+n200; latency consumes stochastic inference RNG.

One-forward smoke (GPU0 must be idle; coordinate with G4 calibration):

```sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /cache/independent_sr_cuda_20261013/env/bin/python \
  /cache/independent_sr_cuda_20261013/code/hypir_cuda_runner/runner.py \
  --model-dir /cache/independent_sr_cuda_20261013/submissions/hypir_s6000_mt200_ct175/model_dir \
  --smoke --out /cache/independent_sr_cuda_20261013/calibration/hypir_s6000_ct175_smoke.json
```

For the calibrated benchmark use the existing `benchmark_cuda.py` with
`--model-dir` pointing to this same model_dir, `--warm 30 --n 200 --device 0`,
and `--ref-ms` equal to the same GPU G4 `event_p10`. Pass `--hash-file` for the
external checkpoint and the three base-model safetensors. The initialization
cost, input allocation and post-timing validation remain outside timing.

`config_s6000_mt200_ct175_fused.json` prepares an optional, unvalidated variant.
It executes `UNet.fuse_lora(scale1,safe_fusing=True)` then `unload_lora` after
the same initialization. BF16 merging changes operation order and rounding;
its benchmark and full quality score need a separate version. No fused run is
started by this implementation task.

Dependencies are owned by the visual-review agent's new env: Torch2.8 CUDA12.8,
torchvision0.23, diffusers0.33.1, transformers4.56.2, peft0.17.1,
accelerate1.10.1, einops0.8.1, numpy2.3.3, Pillow11.3 and safetensors0.6.2.
No software is installed by the Runner implementation.
