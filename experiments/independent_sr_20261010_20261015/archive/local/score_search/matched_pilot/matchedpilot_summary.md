截至 2026-10-14T00:04:04.467172+08:00，三个同输入、同顺序 pilot 的 CPU 复核通过。每版 23 图、98 标量，逐图 PNG/input/GT SHA 与记录一致；逐项均值后代入公式，重算差为零。以下仅为 S_subset，不能与全100图 NR 混比。

| 版本 | NR(test20) | FR(val3) | S_subset | 相对 control |
|---|---:|---:|---:|---:|
| s6000_ct175_compiled_control_pilot | 2.250856000 | 1.463881711 | 3.714737711 | +0.000000000 |
| nr_conservative_s6100_compiled_pilot | 2.259910396 | 1.464764992 | 3.724675388 | +0.009937678 |
| nr_aligned_s6100_compiled_pilot | 2.255283384 | 1.464345475 | 3.719628859 | +0.004891149 |

保守100步版的 NR 增益主要来自 MANIQA 和 CLIPIQA；MUSIQ 均值、NIQE 项略退。FR 增益主要来自 LPIPS，DISTS 略退。三版渲染和评分各四套最终 RNG 字节状态完全相同；8 个实际指标权重 SHA 与原工具压缩包记录一致。CPU 审计没有创建 CUDA 上下文。

旧 compiled full 与新 control 的验证 prefix 不完全字节重现：val1/2 PNG 不同，val3 全等；val1/2 平均绝对差分别为 0.08810/0.05781 个 8bit 级，最大差 15/6。FR 约 -0.00004725 的差异全部来自 DISTS，LPIPS 不变。记录的输入、权重、Runner/renderer source、配置、flags、版本没有发现变化；与 compiled CUDA 浮点/内核/cache 漂移相符，但本次只读审计未证明具体原因。

视觉检查保留 12 原图，45 张原像素 ROI 与来源裁剪逐像素相等。case7/16/23 的材质 ROI 未见保守版较 control 整片规则微纹明确增强，部分细纹略弱。case23 已有蜂窝/点结构仍在。这仅是指定 ROI 的观察，不能证明纹理真实性、全图安全或视觉总优势。

[详细审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/matched_pilot/matchedpilot_audit.json>)；[RNG/权重证据](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/matched_pilot/matchedpilot_rng_weights_audit.json>)；[视觉审查](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/matched_pilot_visual/inspection_roi/visual_review.md>)。
