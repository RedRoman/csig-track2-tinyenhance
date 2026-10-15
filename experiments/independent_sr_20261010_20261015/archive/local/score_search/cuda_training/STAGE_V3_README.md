# CUDA NR v3：6250 FP32冻结anchor与MUSIQ权重分支

两新run均从同一个sealed6250恢复student FP32 LoRA、Adam250及四组RNG，各新增250步（local1..250/global6251..6500/Adam251..500）。旧v1/v2源码与旧run保持只读。teacher先由raw6000初始化并LoRA.float，再逐项精确复制父resume["lora"]，不通过BF16加载6250权重；启动时torch.equal证明514份FP32权重相等并确认teacher冻结。

新objective的保真项以父6250为中心；reward为`musiq_weight*MUSIQ/100 + MANIQA +0.1*CLIPIQA`，总reward_weight0.075、lr2e-7、gradient_anchor0.3、LPIPS权0.1等均保持父值。两个分支仅musiq_weight=1或1.5，STE固定False。这是新的anchor/reward objective，Adam/RNG为显式warmstart，不冒称父原objective连续训练。

新protocol记录v3源码、冻结v2派生源码、v1/helper SHA、父config/resume/protocol链、effective6250 anchor及raw6000初始化路径、teacher FP32 exactproof和MUSIQ权重。student和teacher同起点时首步fidelity预期数值0；仅freshstage首步执行≤1e-6 guard，并记录实际值，再运行三NR真实梯度/LoRA/冻结审计，之后才opt.step。

根任务分配空闲GPU后调用（本agent未启动）：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python -u \
  /cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr_stage_v3.py \
  --config /cache/independent_sr_cuda_20261013/code/cuda_training/config_stage3_anchor6250_m1.json

/cache/independent_sr_cuda_20261013/env/bin/python -u \
  /cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr_stage_v3.py \
  --config /cache/independent_sr_cuda_20261013/code/cuda_training/config_stage3_anchor6250_m15.json
```

run为cuda_nr_stage3_s6250_anchor6250_m1_250与cuda_nr_stage3_s6250_anchor6250_m15_250，各run保存step6300..6500.pth/resume_latest.pt和新协议。仍需独立实际PNG评分判断质量。
