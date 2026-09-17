---
name: project_v17_bridge_sigma
description: V-JEPA-LLVE v17 实验——wanvj 前 16 版一直是 σ=0 确定性直线桥，从未测过 RIVER 的随机桥；2026-08-20 起在 3 台 A100 扫 σ_0
metadata: 
  node_type: memory
  type: project
  originSessionId: 5ebc73ee-9d27-465e-8ded-f9bcfd2c216e
---

2026-08-20 起的 v17 实验。属于 [[project_vjepa_llve]] 主线（CVPR 2027）。

**核心发现（这是起因）**：`tools/v11_wanvj_phase2_v3_train.py` 第 1060 行前向过程是
`z_t = (1-tv)*z_lq + tv*z_hq`，**没有噪声项**。v11 到 v16 十六个版本跑的全是
**确定性直线桥（σ=0）**，即 rectified flow between endpoints，从来没注入过
`σ(t)ε`。RIVER (AAAI'27 投稿, `~/Documents/RIVER_AAAI27/main.pdf`) 3.2 节
"Bridge matching, not rectified flow" 专门用来划界的反例，逐字就是这行代码。
所以 **RIVER 的单一核心 claim，恰好是 wanvj 全部消融史里唯一没测过的变量**。

**为什么值得做**：PROGRESS.md §10.6 的结论是"别再加 adapter，要改训练目标"。
σ 注入正是训练目标级改动。机制假设：σ=0 时每对 (z_lq,z_hq) 只定义一条直线，
训练信号在轨迹邻域是零测集；1-NFE 从 t=0 一步外插必然落在无监督区域 —— 这对
v14 的局部闪烁、v12a 的 seed 方差 ±0.40、pair31 collapse 是一个统一可证伪的解释。

**实现**：`tools/v17_wanvj_bridge_sigma_train.py`，由 `make_v17.py` 从 v3 trainer
生成（带 anchor 断言 + 语法/正则自检）。单变量：`z_t += σ_0·sin(πt)·ε`，
`v_target = z_hq - z_lq` **不变**（对齐 RIVER Eq.3）。σ(0)=σ(1)=0 所以 1-NFE
推理路径逐位不变。`--bridge_sigma 0.0` 精确复现 v10。训练 log 加了 `sg=` 字段
打印实际注入噪声 std —— 见 [[feedback_patch_installed_vs_fired]]，冒烟已验证
σ_0=0.2 时 `sg=0.1959`。

**σ_0 取值依据**（`tools/probe_latent_scale.py` 实测，不是拍脑袋）：
Wan latent 经 per-channel 归一化后 z_lq std≈0.74、z_hq std≈0.65，
**drift std≈0.54，RMS(drift)/RMS(z_hq)=0.83** —— transport 极大，不是小残差。
clip 间 drift 散布 5 倍（0.19~0.95）。据此扫 σ_0 ∈ {0.1, 0.2, 0.4}。

**基准**：v10-seed0 final，11-seq PSNR **21.578** / SSIM 0.7664 / LPIPS 0.2422 /
DISTS 0.2343 / CLIPIQA 0.2396。完整启动参数从
`wanvj_phase2_v10_seed0/ckpt.pt` 的 `args` 键里恢复，已固化进
`/temp/zihaohe/launch_v17.sh`。注意 v10 的 `lambda_l1=0`、`lambda_chroma=0`，
只有 lpips=0.1；`dove_prob=0` 所以 DOVE 分支不触发。

## 第一批结果（2026-08-21 出，单 seed，11-seq held-out）

| run | PSNR | SSIM | LPIPS↓ | DISTS↓ | CLIPIQA↑ |
|---|---|---|---|---|---|
| v10-seed0（历史基线 σ=0） | 21.578 | 0.7664 | 0.2422 | 0.2343 | **0.2396** |
| σ=0.1 | **21.698** | 0.7662 | 0.2436 | 0.2361 | 0.2224 |
| σ=0.2 | 20.983 | **0.7682** | 0.2510 | 0.2432 | 0.2234 |
| σ=0.4 | 21.351 | 0.7674 | **0.2403** | **0.2342** | 0.2356 |

**结论：什么都没证明。** PSNR 差 +0.12/−0.60/−0.23，非单调，且 +0.12 远小于
PROGRESS.md 记的 seed 方差 ±0.40。σ=0.2 出现 v15 式 pair31 collapse
（25.33→21.25），可放弃。σ=0.1 恰好同时满足 §10.7 的两条成功标准
（pair31 只 −0.16 在 −0.2 容差内；pair15 +2.13、pair33 +2.13），
但 **CLIPIQA 掉 0.017、LPIPS/DISTS 都变差** —— 即使为真也是拿感知换保真。

`sg=` 已核：σ=0.1 实测 mean 0.0676 / max 0.1002，理论 0.637×0.1=0.0637 / 0.1，
噪声按设计注入（[[feedback_patch_installed_vs_fired]] 的检查过了）。

## 红旗：pair13 剂量无关地掉 2.2 dB

三个 σ run 的 pair13 分别 −2.25 / −2.26 / −2.18，**与 σ 剂量完全无关**。
剂量无关的效应不可能由剂量引起 —— 说明 v17 run 与历史 v10-seed0 之间
除 σ 外还有一个固定差异（主机/cuDNN、数据改从 /temp NFS 读、
或注入的 `randn_like` 改变了全局 torch RNG 流）。
**所有 σ 的 Δ 都算在了可疑基准上。**

## 第二批（2026-08-22 01:47 起，host 30674 两卡，各约 8h）

- GPU0: **σ=0.0 seed 0** —— 同源控制组，专门回答 pair13 红旗。
  若复现出 21.578/pair13≈25.5 则环境无辜；若给出 pair13≈23.3 则基准要重算。
- GPU1: σ=0.1 seed 42 —— 给处理组第二个 seed。

跑完即有 2 seed × 2 条件，才第一次能对 "σ=0.1 vs σ=0" 的差值给误差棒。
日志 `/temp/zihaohe/cache/checkpoint/v17_sigma00_seed0.log`、`..._sigma01_seed42.log`。
主机环境见 [[reference_modelarts_shared_hosts]] —— 30313/32702 的卡在我们的 run
退出后立刻被别的租户抢走了，跑完要马上占坑。

**顺带发现的坑**：PROGRESS.md §10.7 提的 v17 residual 方案有塌成 no-op 的风险 ——
现在 `v_target = z_hq - z_lq` 且 1-NFE 时 `z_out = z_lq + v_pred`，模型**已经**在
预测残差并加回 LQ。除非把源分布换成高斯，否则那只是同一函数的重参数化。上机前先纸上推。
