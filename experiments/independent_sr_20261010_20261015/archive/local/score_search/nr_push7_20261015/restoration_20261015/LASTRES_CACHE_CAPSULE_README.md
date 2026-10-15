# Last-residual six-parameter checkpoint capsule — cache/Mac only

This package protects the exact original bytes of `step7800.pth`, `resume_step7800.pt`, `step7850.pth`, and `resume_step7850.pt` from `hypir_push7_A7750_lastres_nr100`. All 514 LoRA tensors are saved; only six final residual/output tensors were optimized, while 508 remain exactly A7750. Original marker SHA values bind all four files. CAS blobs preserve changed tensor/Adam/RNG/pickle/header bytes; common payload ranges reference the immutable A7750 resume.

This package is stored only under `/cache` and on the Mac. It does not create or modify an SFS prefix. Its payload limit is160MiB. Existing v1/v2 SFS archives remain frozen, and no 4GB checkpoint archive is downloaded.

Required persistent reference: `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/resume_step7750.pt`, exactly3,114,332,685 bytes, SHA256 `e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b`. It is not duplicated. This is a reference-based recovery package rather than a self-contained copy of every tensor.

After copying the small package back to `/cache/push7_lastres_nr_capsule_20261015` on a new cache, restore and verify with standard Python only:

```sh
python3 /cache/push7_lastres_nr_capsule_20261015/metadata/code/lastres_zip_capsule_cache_only.py verify
```

The output directory `/cache/push7_lastres_nr_capsule_verification_20261015` must be new. The helper validates the full persistent base SHA, compressed and decoded payload SHA, every reconstructed target, and then rereads the four complete disk files to compare their original marker SHA. Check the downloaded manifest and DONE hashes before recovery. Raw `.pth` is for model loading; use its matching full `.pt` resume for exact training continuation.

Training source/config,100-step log, milestone markers, original input SHA catalog, small source files, root-registration provenance and minimal CPU log/RNG proof are included. This package proves checkpoint recovery, not a quality or target-score result. No GPU, distillation, validation-GT training, pointer or historical-score change is performed by packaging.
