---
name: project-vosr2-fill2sr-recon
description: VOSR2（2026-08-22 发布，单步 1.4B DiT）教师候选调研结论与 Fill2SR 不可用的原因
metadata: 
  node_type: memory
  type: project
  originSessionId: 104e347e-c0e8-47a0-8633-b9d5b60702af
---

2026-08-30 调研（CSIG 赛道二教师候选）：

**VOSR2 可用且高度相关**：官方 VOSR repo 2026-08-22 发布 VOSR 2.0——
单步 1.4B DiT、vision-only、主打 fine-structure（小脸/小字/建筑/植被）。
下载：ModelScope `LULALULALU/VOSR_CKPT` 子目录 `VOSR2/`（ema_model.safetensors 5.58G）。
推理：`inference_vosr_onestep.py -c preset/ckpts/VOSR2`，用 Qwen-Image-vae-2d
（30674 的 /cache/zihaohe/VOSR 已有该 VAE，repo 已 git pull 到 d72c64d）。
**2026-08-30 试点已完成,VOSR2 胜出**（200 张确定性 mix-LQ 同场）：
FR: LPIPS 0.1815≈PixelDiT 0.1784,DISTS 0.1140 全场最佳（VOSR 0.181/PixelDiT 0.149）,
强档几乎无退化（0.1815→0.1863,PixelDiT 0.178→0.240）;lf_top 0.0214 最干净。
NR 合计 2.32:比 PixelDiT 2.05 高 0.27 但低于 GT 2.63、远低于 VOSR_v3 3.16——
预测学生 NR 仍会小幅塌,需学生侧 IQA 补（v8c 已证可行）。
已启动 20k 全量教师生成（30674 GPU0,2.45it/s）+ v9_relay 自动接力
distill_v9（v8b 同配方仅换 teacher,单变量）。
接入坑:inference_vosr_onestep.py 的 torch.hub.load 即使有缓存也要访问 github,
已 patch 成 source='local' 走 preset/ckpts/torch_cache/facebookresearch_dinov2_main。

**Fill2SR 不可用**：ECCV 2026 poster（9-12 展出，比赛 9-15 截止），
作者 Xingfu Yi / Xiaoxue Yu，无 arXiv 无代码无 checkpoint，只有标题
（把 inpainting DiT 改造为 Real-ISR，思路类似 FLUX-Fill 系）。不必再追。

相关：[[project-csig-track2]] [[project-pdit-teacher-pilot]]
