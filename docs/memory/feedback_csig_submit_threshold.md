---
name: feedback-csig-submit-threshold
description: CSIG 赛道二剩余提交只在预测榜分接近 150 时才考虑；不要拿 140 上下的包去提交读分
metadata:
  type: feedback
  originSessionId: 09b1d3
  modified: 2026-09-18T13:05:00.000Z
---

2026-09-18 用户指示：L2d_g 包（预测约 142）"先不用，大致到 150 左右才考虑下一次提交"。

**Why:** 余 4 次提交、截止 09-20，用户目标 150；每次提交都要冲着目标设计，不做挤牙膏式读分。
中间档的包只是保底，不值得消耗次数。

**How to apply:** 训练/打包照常推进，但只有当"test NR + val FR 代理分 + 实测时延"代入四点拟合模型
给出约 150 时，才把包拉到本地并向用户报"可提交"。低于此只汇报进展，不催提交。
保底包 `~/Downloads/赛题二_pixL2d_g_5e24ed6f.zip` 留着，截止前若无更好候选再由用户决定。

相关：[[project-csig-track2]] [[feedback-eta-on-launch]]
