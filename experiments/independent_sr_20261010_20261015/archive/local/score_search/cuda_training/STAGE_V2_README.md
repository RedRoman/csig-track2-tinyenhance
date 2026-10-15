# CUDA NR stage v2（6250→6500）

新阶段local1..250，对应global6251..6500，Adam step251..500。旧v1源码、旧run、父resume只读；v2自身及量化helper的SHA进入新protocol。父resume/config/protocol/旧source全部显式SHA钉定。

从raw6000初始化、LoRA升FP32，然后从父resume逐值复制FP32 LoRA并做精确相等核验，再恢复AdamW及四组RNG。初始化、指标构造、所有metadata检查结束后才恢复父RNG。没有从BF16重载step6250.pth替代FP32状态，也没有新的Adam trajectory。

无STE分支保留原objective；roundSTE分支是明确的新objective，父Adam/RNG作为有记录warmstart。两分支保留原6000/coeff175 frozen同latent anchor、原超参、真实NR梯度、训练MANIQA20/正式100；每50步保存raw LoRA和resume。首步在opt.step之前执行既有三NR梯度/LoRA/冻结检查。

使用下列两份配置（由根任务分配GPU0/1；本agent未启动）：

```sh
CUDA_VISIBLE_DEVICES=0 /cache/independent_sr_cuda_20261013/env/bin/python -u \
  /cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr_stage_v2.py \
  --config /cache/independent_sr_cuda_20261013/code/cuda_training/config_stage2_cont250.json

CUDA_VISIBLE_DEVICES=1 /cache/independent_sr_cuda_20261013/env/bin/python -u \
  /cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr_stage_v2.py \
  --config /cache/independent_sr_cuda_20261013/code/cuda_training/config_stage2_roundste250.json
```

run分别为cuda_nr_stage2_s6250_cont250、cuda_nr_stage2_s6250_roundste250。新run必须尚不存在；将写parent_migration.json/config.json/gradient_preflight.json/train.jsonl/step6300..6500.pth/resume_latest.pt。阶段内恢复需显式--resume本阶段resume_latest.pt，严格匹配新protocol，仍保持阶段最多250步。

当前仅完成CPU静态编译与源码审查；不宣称新训练已开始。精确PNG surrogate的CPU bitexact/identity-gradient审计由live_progress提供。
