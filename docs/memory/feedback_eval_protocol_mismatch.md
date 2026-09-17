---
name: feedback-eval-protocol-mismatch
description: 比对两个数字前先确认它们是同一口径产出的，否则会把口径差误判成退化或提升
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8e5b7e1d-72f5-4289-ad74-10d03f4df098
---

CSIG 项目里训练 `log.json` 的 `score` 是训练中的 **crop 验证**口径（`--val_crops`），
而 `eval_vosr.py` / `eval_ck.py` 输出的 `[验证集全尺度]` 是**原生 4K 全尺度**口径。
两者数值差得很远（vosr_cap：crop 4.2787 vs 全尺度 4.1095）但**不可比**。

**Why**：这类错误已经反复出现——2026-07-22 Seer/AAAI27 "差 2.7dB" 查明是测错了
checkpoint+协议组合；2026-08-13 查明全部本地评测在 512 crop 而官方在原生 4K，
调优全白做。口径不匹配产生的差值看起来像真实的性能变化，非常有迷惑性。

**How to apply**：拿两个数字做结论前，先回答"这两个数是同一个脚本、同一个协议跑出来的吗"。
验证迁移/复现是否正确时，**不要拿训练 log 当基准**，要用同一评测脚本在两端各跑一次对数
（2026-08-19 迁移 AutoDL 就是这么验的：同 ckpt 同脚本，VOSR 的 Score 3.8957 与 NR 2.4733
两端逐位相同，才算真通过）。历史 log 里已有的同脚本记录可以直接当黄金对照。

相关：[[feedback-verify-on-holdout]] [[project-csig-track2]] [[project-seer-aaai27-repro-gap]]
