---
name: project-csig-hallucination
description: "CSIG赛道二：高NR模型会凭空生成仙人掌状物体，NR优势=幻觉，\"加强教师\"路线因此作废"
metadata: 
  node_type: memory
  type: project
  originSessionId: 104e347e-c0e8-47a0-8633-b9d5b60702af
  modified: 2026-09-14T02:35:42.296Z
---

2026-08-20 目视确认（`montage.py` / `montage_tinysr.py`，输出在 host `/cache/zihaohe/montage_*.png`）：

**VOSR 族微调后会在楼体/楼顶凭空生成整片仙人掌、多肉状绿色物体，还有黑色旗状物。**
不是纹理伪影，是**整体物体幻觉**。程度随 NR 单调加重：
- `vosr_teacher2`（NR 3.293、MANIQA **1.1032 越出[0,1]值域**、PSNR 18.06、LPIPS 0.6661）：整片仙人掌
- `vosr_cap`（NR 2.901、Score 4.1093）：楼顶一小簇，同病较轻

**TinySR 族干净**（run12 / mosaic_cap / distill_v1 均无物体幻觉；mosaic_cap 有斜向编织纹）。

**关键推论**：教师 vosr_cap 的 NR 2.901 → 学生 distill_v1 只拿到 2.768，
测试集 NR 保留率 40%。**这不是蒸馏失败，是 GT 兜底项 + FR 势垒正确挡住了幻觉的抄袭。**
教师比学生多出的 NR 就是幻觉贡献的，所以「加强教师再蒸馏」这条路作废——
推得越狠幻觉越重，而学生根本不会继承。

**因此**：此前算的「NR 上限 3.4 → 折合 155 分」是虚的，那个区间里全是幻觉。
真实可用上限必须在「视觉无幻觉」约束下重新标定。
优化目标必须加一条不可协商的约束：不得产生物体幻觉。

赛题原文：「指标异常高但主观明显不合理 → 取消资格」，人工复核在决赛阶段。

教训：**指标能被攻击，眼睛不能。** 这是本项目第七次「拿窄观测面下结论」，
前六次记在 [[feedback-verify-on-holdout]]。相关：[[project-csig-track2]]

**2026-09-14 像素 UNet 的"眼睛"幻觉(第二种形态)**:moonshot 学生 pixB2(11M ps2 像素 UNet)
在 v8 配方(w_iqa 2.0/maniqa 3.0/clipiqa 2.0)下全尺度 S=4.28、MANIQA 0.79、CLIPIQA 0.95——
全是假的:树冠/窗户/墙面长满带瞳孔的圆形"眼睛"斑(DeepDream 式)。机制:无 VAE 先验的像素网
在 IQA 直推下直接生成 MANIQA/CLIPIQA 的对抗纹理;NIQE 反向变差(3.33→3.97)是唯一指标侧警报,
lf_top 抓不到(0.077 vs 0.073,它只测低频偏移)。同模型 IQA 开启前(pixC@8000)干净但发软 S=3.38。
**规则:CLIPIQA>0.9 或 MANIQA 比 v8 高 0.05 以上的结果必须先看图再谈分数。**
对策:w_iqa≤0.5、MANIQA/CLIPIQA 权重砍 3-4 倍、NIQE 加倍当自然性正则、纹理靠 GAN(教师=真样本)
与对教师的高频 L1 传,不靠指标直推。对比图 ~/Downloads/pix_viz/(LQ|pixB2|v8|GT)。
