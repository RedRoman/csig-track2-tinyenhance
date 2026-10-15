The deployed entry is `/cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr.py`.
This task prepared and statically compiled code only; root owns GPU1/GPU2 launch.

Two independent100-step pilots warm-start the raw, unfused step6000 checkpoint
SHA `2c82a71a94aa754bd4a26a2a474c2bbe4989bde1d38d2e15d9e58eb0c3e408bc`.
Both use the same frozen checkpoint/coeff175 anchor and exact detached student
latent. The base model stays BF16; only FP32 rank256 LoRA parameters update.
Both use a new AdamW (weight_decay0, foreachFalse) and seed20261013, reset after
all metric constructors because pyiqa constructors reset the seed to123.

The original pre6000 training used batch1/patch512/sample, lr5e-7, reward0.1,
clip0.1 and spatial-gradient anchor0.1. Historical corrective segments used
lr2e-7, reward0.075 and gradient0.3. The old MANIQA helper used4 crops;
new pilots explicitly use20 training crops, while formal scoring remains100.
MUSIQ/MANIQA/CLIPIQA are the actual supplied FP32 networks, frozen/eval with
as_loss=True. Hidden CLIP RN50 is separately frozen. NIQE stays evaluation-only.
LPIPS/L1/spatial-gradient fidelity use unquantized RGB. The current configs
disable the optional round STE; native quality remains an independent PNG test.

Training matches the deployment input's FP16[-1,1] roundtrip, then applies the
original HYPIR wavelet reconstruction to both student and same-latent teacher.
The first step audits three nonzero/finite NR input gradients and finite,
nonzero LoRA gradients before any optimizer update. An initial zero fidelity
loss is valid because teacher/student initialize identically.

```sh
CUDA_VISIBLE_DEVICES=1 /cache/independent_sr_cuda_20261013/env/bin/python -u \
  /cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr.py \
  --config /cache/independent_sr_cuda_20261013/code/cuda_training/config_conservative.json

CUDA_VISIBLE_DEVICES=2 /cache/independent_sr_cuda_20261013/env/bin/python -u \
  /cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr.py \
  --config /cache/independent_sr_cuda_20261013/code/cuda_training/config_aligned.json
```

Conservative: lr2e-7/reward0.075/clip0.1/gradient0.3.
Aligned: lr1e-7/reward0.05/clip1.0/gradient0.3. Models use model_t200/coeff_t175.
New output directories are `adaptation/runs/cuda_nr_s6000_ct175_*_pilot100`.
Each saves raw LoRA at6050/6100 and a transactionally saved `resume_latest.pt`
with optimizer and Python/NumPy/CPU/CUDA0 RNG. Resume requires explicit
`--resume`, identical algorithm/input/source/weight identity and a compatible
target step. Existing old runs/weights are never overwritten. No extra public
adapter or metric weights are needed for the chosen current6000 anchor.

Optional one-step preflight: append `--preflight-only --preflight-case23`.
It performs real loss/backward audits but never calls opt.step or saves weights.
Running a separate preflight is unnecessary: the training entry does the same
audit before the first real update.

Both100-step pilots have now completed. Each training loop took about95 seconds,
with a steady median of0.82seconds/step and peak allocation27.85GB. A250-step
loop should take about4minutes at this rate, plus fresh model/metric startup;
this estimate excludes independent render/scoring. Selection still depends on
the matched pilot20+validation3 quality results, not training-crop reward alone.

The initial parameter precision is explicitly `raw FP32 -> plain HYPIR BF16
load -> FP32 trainable`. It matches the existing deployment's checkpoint load
but is not a bitwise FP32 optimizer continuation from the old NPU run. The
independent CPU audit separates initial quantization from genuine AdamW updates.
Relative to that actual starting point, conservative/ aligned parameter L2
changes are0.0458376/0.0253364. After loading the new weights into BF16 for
deployment,17.039%/9.880% of values still differ from the initial deployed
weights. Both differences are genuine parameter updates, not serialization SHA
changes. Both branches used identical100-case/crop sequences and finished with
identical Python/NumPy/TorchCPU/CUDA RNG states. All514 LoRA gradients were
finite/nonzero on every step; teacher/metric gradient guards passed.

`resume_latest.pt` contains FP32 LoRA values exactly matching `step6100.pth`,
and each AdamW parameter state is at step100. On a future explicit resume, the
entry copies these saved values into the already-FP32 parameters after initial
loading, then restores AdamW and all RNG states. This preserves the new CUDA
run's FP32 optimization continuity. Source/config identities of completed runs
were left unchanged; these notes are a sidecar explanation.


Latest completed status (2026-10-14, sidecar update; frozen training source/protocol unchanged):

Conservative explicitly resumed from local100/global6100 to local250/global6250. Both full evaluations are complete103 images /418 scalars and passed CPU actual-PNG/input/GT/source/checkpoint hash and mean-formula audits. Full compiled6000 / conservative6100 / conservative6250 quality S_proxy values are3.717100038686201 /3.725682022953923 /3.7301015304490415. Pilot20+val3 values remain separate S_subset records. The250 checkpoint has genuine FP32 updates relative to6100; resume_latest LoRA equals6250 and all514 Adam states are at step250. Final CUDA latency/tool predicted total and SFS backup are pending in the main task.

step6250 SHA256 `7b57349706372122ea67173fbb944e7e806d86e12379e3fd66ea2ebf1f96ecc3`; resume_latest SHA256 `3dcf896028f41c7c4c6e15074cc69c4cdb2ff7679e6756f5ee57a9a657f61c1f`. The directory still ends_pilot100 but latest completed target is250. The original100-step config/status are preserved in config_step6100.json/status_step6100.json.

[Chinese completed experiment report](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_FINETUNE_REPORT_20261014.md>); [250 CPU parameter/optimizer audit](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/training_250_parameter_audit.json>).

Latest measured status (2026-10-14; supersedes prior pending latency note):

Fresh singleA100 GPU0 warm30+n200 eventp10 timings are46.43932037353515ms /46.62579345703125ms /45.78047866821289ms for compiled6000/conservative6100/conservative6250. With the same-card G4ref0.6502400040626526ms, unmodified original predict.py estimates54.861316549544334 /55.144742603774404 /55.51522024624155. This five-point pixel-model empirical prediction extrapolates toHYPIR and is not an official leaderboard result or a guaranteed±2point interval. Small timing differences are not proof of algorithmic acceleration. Thirty-one downloaded small records are byte-hash-identical to remote; both trained versions have byte-identical val3prefix betweenpilot andfull. Main task owns the final SFS backup proof and report sync.

Final archive status: verified core SFS backup complete (670files /18,543,300,225bytes), including both exact optimizer/RNG resumes and selected checkpoints. Final metadata refresh follows in the shared manifest. [Restore guide](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/RESTORE_NR_CUDA_20261014.md>). Both training branches and all render/score/benchmark GPU jobs have exited normally.
