---
name: project-pdit-teacher-pilot
description: PixelDiT@240k 在我方 mix 域上全面碾压 VOSR 教师，已启动全量 20k 教师生成换教师蒸馏
metadata: 
  node_type: memory
  type: project
  originSessionId: 104e347e-c0e8-47a0-8633-b9d5b60702af
---

2026-08-27 试点（200 张确定性 mix-LQ，与 teacher_v3_mixlq 同 idx 同场）：
PixelDiT@240k：LPIPS 0.178 / DISTS 0.149 / PSNR 25.04 / lf_top 0.025
VOSR teacher_v3_mixlq：LPIPS 0.241 / DISTS 0.181 / PSNR 19.66 / lf_top 0.071
强档子集（91 张）PixelDiT 同样全胜（PSNR +4.4dB，lf_top 干净 2.6 倍）。

结论：R2 反馈里"真实域 PSNR 转负"不影响它当**合成 mix 域上的 FR 教师**——
蒸馏对就在这个域上。FR 轴是学生最大缺口（全尺度 lpips 0.354/dists 0.303），
每降 0.01 直接 +0.01 Score。

已启动：20k LQ（lq_mix05，导出与训练端 rng byte-exact 验证 14/14）+ GT（gt_full）
传 AutoDL，GPU2/3 双卡各 10k 分片推理（0.9s/张实测）。产出后挂 distill_v8b =
v6 配方 + teacher_dir 换 PixelDiT，单变量对照。SR 文件名 {seq}_{idx}_sr.png 需重命名。

相关：[[project-csig-track2]] [[reference-csig-hosts-layout]]
