---
name: reference-csig-hosts-layout
description: CSIG 双机布局——蒸馏主线在 ModelArts 30674 本机，AutoDL 经 30674 跳板 ssh autodl
metadata: 
  node_type: memory
  type: reference
  originSessionId: 104e347e-c0e8-47a0-8633-b9d5b60702af
  modified: 2026-09-16T14:58:37.091Z
---

2026-08-27 实况（纠正 [[reference-autodl-csig-host]] 的过时印象）：

**蒸馏主线在 ModelArts 30674 本机** `/cache/zihaohe/`（MCP host_id 用 `"30674"`，
不是 "huawei-modelarts-30674"）：train_distill.py、eval_ck.py、csig_isr/train_out/
（distill_v6=4.1705 等）、distill_data/（教师输出集）。2×A100-80G 共享，
别的租户常驻 30-45GB，用 nanny.sh 排队。

**AutoDL 只能从 30674 跳**（`ssh autodl`，本地 Mac 直连会 publickey 拒绝——
本地没装对应私钥）。AutoDL = 4×vGPU-48G：GPU0/1 常被同事 songhua 的
PixelDiT/DMD2 训练占（dmd2_r5 等）；zhuyan_work 目录是另一位队友的。
我方在 AutoDL 的落脚点 `/root/autodl-tmp/zihaohe/`。
PixelDiT 推理入口：`eval_checkpoint_paired_lq`（repo 在
pixeldit_controlnet_continue_200k/PixelDiT，交付 ckpt 在
pixeldit_controlnet_csigv2_detail_losses5x/delivery/olddeg_init_current，
env 用 songhua 的 pixeldit-sr venv，须 --fixed-prompt 否则 captioner 报错）。

**坑(09-16)**:在 `ssh autodl '... pkill -f "[t]rain_pix.py"; ...'` 这种单引号命令串里,pkill 的 -f 模式会
匹配到**承载它的那条 ssh/bash 命令行本身**(命令行含该字面串),把后续所有 launch 一并杀掉且无任何报错——
表现为"launched: N"行凭空消失、日志为空。要 kill 训练请用 `pgrep -f "train_pix.py --arch"` 这类不出现在
外层命令里的模式,或分两次 ssh。另:AutoDL GPU0/1 上常有队友 `independent_sr`(diffbir)进程,勿杀。

相关：[[project-csig-track2]] [[project-pdit-teacher-pilot]]
