# Push5 HYPIR inference speed review (CPU preparation only)

Parent: `push4_a7750_compiled_all`, selection SHA `b3148b21f466854f9fe8f81c7eca1348da5bf793141799573ff3be8b20e25c3f`.
Frozen runner SHA `396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae`.
Checkpoint unchanged: A7750 raw SHA `589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d`.

## Budget required by the unchanged prediction tool

With S=3.7614565663241937 and the Push4 fresh G4 event p10=.7813119888305664ms,
70 would require normalized g=10.466412891380699ms, or A100 event p10=18.973396454957154ms.
Current A7750 is46.34193916320801ms. This requires2.4425x throughput at fixed quality.
These thresholds are conditional on that same G4 calibration, not a prediction of any new candidate.
The new candidate is unmeasured at preparation time; an engineering estimate of1.2–1.5x is plausible,
but there is no evidence for2x or70 from this route. Root is also pursuing a faster G4 student.

## Source findings

- Empty prompt text embedding and timestep are already cached; no per-call text encoder remains.
- LoRA is already fused/unloaded and UNet/VAE encode/decode already use default compile.
- Native image processing still runs Python tile loops, posterior sampling and DDPM scheduler.
- The external make_tiled_fn performs Gaussian multiply, BF16 output accumulation and FP32 count division even for one tile. Skipping these changes BF16 rounding and cannot reuse parent quality.
- AutoencoderKL internal tiled VAE is not enabled by BaseEnhancer.init_vae. The relevant512 tile work is the external tiler.
- DDPM.step computes prev_sample and variance although caller takes only pred_original_sample. Omitting unused arithmetic may help little; the variance randn call must remain in its original position to preserve four-RNG progress. No scheduler change is included.
- Wavelet color reconstruction performs ten full-resolution FP32 depthwise convolutions, plus repeated constant creation and unused style-high-frequency arithmetic. Its cost must be measured; no component latency claim is made from asynchronous tqdm timings.
- BF16 network precision is retained. TF32 toggles concern FP32 matmul/conv; matmul TF32 is currently false and cuDNN TF32 true. Enabling flags is not assumed to significantly accelerate this BF16 network.
- FP16 conversion, replacing Gaussian stitching, scheduler algebra changes, separable/custom wavelet kernels and whole-image CUDA capture are deferred. Any changed outputs need an independent formal full103 evaluation.

## One minimal candidate

`/cache/independent_sr_cuda_20261013/submissions/hypir_push5_a7750_clro_wavelet/model_dir`

Runner SHA `34a70fae57bc412fd3e0394e0573942e63c2764334847dab67eee3c4b696b092`.
Config SHA `eb8f5cbd85adf85f97f8ee3783f605fb766e96b8d631b7ad04db45078b036b66`.
Independent candidate_manifest.json is alongside model_dir.

Changes: channels_last parameter storage for UNet/VAE before compile; reduce-overhead for UNet.forward,
VAE.encode, VAE.decode, and complete wavelet_reconstruction; cudagraph_mark_step_begin once at each infer entry.
All image sizes use the same algorithm and deterministic subgraph choices. No benchmark/data/call-count branch.
No hidden warmup, input cache, output cache, seed reset or metric modification.
Original input/output transforms, Gaussian tiling, posterior.sample and DDPM RNG calls remain.

PyTorch2.8 describes reduce-overhead as CUDA graphs to reduce Python launch overhead, without a guaranteed gain;
max-autotune is a separate kernel-search option with potentially substantial preparation cost.
[Official compile documentation](https://docs.pytorch.org/docs/2.8/generated/torch.compile.html).
Raw whole-network capture was not added: it would also need cached host-created constants, proper graph-safe RNG
and persistent input/output ownership. Graph replay uses fixed addresses and omits CPU work.
[Official CUDA graph constraints](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs).
Installed Torch source was independently read at
`/home/ma-user/miniconda3/lib/python3.13/site-packages/torch/_inductor/__init__.py`
and `torch/cuda/graphs.py`; reduce-overhead sets triton.cudagraphs=True.

## Validation commands for root scheduling

No GPU was launched by the preparation agent. Root chooses CUDA_VISIBLE_DEVICES.
The unchanged benchmark must be used in a fresh process (warm30/n200/seed231), with the exact raw checkpoint --hash-file.
Do not substitute historical timings into a new official prediction; fresh same-card G4/parent/candidate are needed for that.

Diagnostic helper:
`/cache/independent_sr_cuda_20261013/code/push5/inference/compare_push5_forward.py`
SHA `83dff1404ad08b4be470688432a8411df0ebd4a08bfe25af000d9e8a53596d95`.

Run --phase reference then candidate in separate processes, --suite smoke512 then native7_41, with a shared
fresh --out-dir for each suite and the corresponding --model-dir.
Checks three distinct512 inputs, native cases7/41, raw FP32 finite/range, PNG bytes, initial/post-call four RNG,
and that retaining previous output tensors does not let later calls overwrite them.
Native reference temporary files consume about200MiB. No full-quality or speed claim follows from this diagnostic.
If timing is not competitive, stop this branch. If outputs change and speed is worth pursuing, render/rescore a new full103 cohort.
