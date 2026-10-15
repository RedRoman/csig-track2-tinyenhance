# LR5e-6 lastres exact recovery capsule

This new package is CACHE/Mac only. It does not write SFS, modify earlier capsules, download the four large checkpoints, or change the selected model.

Run: `hypir_push7_A7750_lastres_lr5e6_nr100`. Fresh Adam updates the six final-resnet/output LoRA tensors; the remaining 508 tensors equal A7750. The metadata includes the frozen source/config, actual 50/100 markers, run logs, input catalog, registration manifests, minimal log/RNG audit, and matched 23-image audit. The matched result is a subset result, not a full-score prediction.

The recipe retains every ZIP header/gap/tail and unique payload. Shared parameter payloads reference this existing immutable SFS file, which is **required** and is not copied:

`/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/resume_step7750.pt`

Base size: `3114332685` bytes. Base SHA256: `e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b`.

Raw target SHA256 values, also bound to the actual saved markers:

- Step7800: `dc184b14f549fc01aeafbdc3bdcf6a7112e6bc3944d6c047c557ed850074d7b7`
- Step7850: `23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c`

After restoring this capsule to `/cache/push7_lastres_lr5e6_nr_capsule_20261015`, run on CPU with the fixed SFS base accessible:

```sh
python3 /cache/push7_lastres_lr5e6_nr_capsule_20261015/metadata/code/lastres_lr5e6_zip_capsule_cache_only.py verify
```

The verification directory `/cache/push7_lastres_lr5e6_nr_capsule_verification_20261015` must not exist. The helper never overwrites it. Verification reconstructs four files there, rereads each complete file from disk, and requires exact original size/SHA before writing a PASS proof. A second verification is unnecessary when the existing proof and committed capsule hashes have already passed.

`manifest.json` and `DONE` commit the small package only after the actual four-file restoration succeeds. The local verifier checks every downloaded file, the exact inventory, compressed and decoded CAS hashes, packaged metadata hashes, marker bindings, and the recorded restoration proof. The helper has no SFS commit command; its only commands are `build`, `verify`, and `package`.
