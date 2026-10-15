TensorRT VAE backend probe is closed without promotion.

| Component | Precision | Torch RO p10 (ms) | TRT p10 (ms) |
|---|---|---:|---:|
| encoder | bf16 | 9.720 | 34.532 |
| encoder | fp16 | 9.666 | 7.555 |
| decoder | bf16 | 18.945 | 72.653 |
| decoder | fp16 | 19.091 | 14.056 |

These are fixed-tile diagnostic component timings, not the original complete Runner benchmark. Dedicated non-default-stream timing was 34.544 ms encoder / 72.618 ms decoder, with byte-identical tensors to the same engines on the caller stream. Actual IO was contiguous LINEAR. The layer profile confirms that kernel execution dominates; detailed precision/tactics were not available.

At the last same-GPU reference (0.649216 ms), preserving A7750 S=3.7614565663 requires complete inference ≤22.989 ms for65. The two FP16 component savings sum to about7.1 ms, much less than the18.45 ms required; this sum is only a rough upper-level estimate, not a measured Runner result. FP16 also changes arithmetic, and neither engine precision was scored on103 native outputs. No A quality reuse or new model score is claimed.

Six small result JSONs and actual source SHA records are local; no engine binaries were downloaded. Original models/environments and archives are unchanged; isolated TensorRT assets remain remote. Root stopped additional builds and Runner work.
