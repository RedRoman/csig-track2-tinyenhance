# Last-conv LR2e-7 exact byte recovery capsule

This capsule preserves the original bytes of `step7800.pth`, `resume_step7800.pt`, `step7850.pth`, and `resume_step7850.pt` without storing four redundant tensor archives. Every target ZIP local header, padding, descriptor, pickle and central-directory byte is preserved through gap blobs; shared tensor payloads reference immutable A7750 resume ranges. Unique payloads use SHA-addressed zlib blobs. Training remains HYPIR only, with two Adam tensors and 512 tensors exactly equal to A7750.

The required existing base is `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/resume_step7750.pt`: 3,114,332,685 bytes, SHA256 `e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b`. It is referenced, not duplicated. The old A7750 raw recipe/helper is also included as small metadata. No CUDA, Torch, model, metric network or unpickling is used for ZIP recovery.

To recover after a new empty cache, copy this small archive to `/cache/push7_lastconv_nr_capsule_20261015`, then run:

```sh
python3 /cache/push7_lastconv_nr_capsule_20261015/metadata/code/lastconv_zip_capsule.py verify
```

The verification directory `/cache/push7_lastconv_nr_capsule_verification_20261015` must be new. The helper first rehashes the complete persistent base, verifies all compressed blobs and referenced ranges, reconstructs four files, then independently rereads each complete restored file SHA. Expected whole-file SHA values are in `capsule.json` and `verification_proof.json`; compare them before restoring files to their original run paths. Use raw `.pth` for model loading and the matching full `.pt` resume for exact training continuation.

This archive proves exact checkpoint recovery and real training provenance; it makes no full-quality or target-score claim. Existing SFS archives and model/score pointers are unchanged.
