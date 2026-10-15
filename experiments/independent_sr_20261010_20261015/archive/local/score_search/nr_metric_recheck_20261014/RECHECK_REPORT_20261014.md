# 最新NR模型指标与可视化复测（2026-10-14）

服务器端口30416，最新保存的训练断点仍为保守分支step6250。此次重新测量已有完整输出的指标，同时下载原4K可视化；没有新增训练或重渲染图片。

独立复测使用用户原始score_dir.py，同一组输入/GT、原100-crop MANIQA和每split连续RNG，分别在空闲的A100 GPU0/1处理6250/6000完整103张。原评分脚本、度量forward、旧结果及原PNG保持不变；新结果另存metric_rechecks/20261014_original_tool。两组各103张的独立复测已完成，共206次图像度量、836个逐图标量；原工具汇总NR/FR/S与上次结果完全一致。CPU最终审计确认两组各418/418逐项数值完全复现，最大绝对差0；实际PNG/103输入/3GT/8指标权重与冻结源码SHA全部通过。

既有完整评分NR：6000基线2.253171080906，6250微调2.266385043359；S分别3.717100038686与3.730101530449。按原工具逐图NR对照，86/100张提高、14张下降，逐图中位增量0.01805638；总体NR仍按各项均值后clamp的原公式计算。

选取case7（文字）、16（建筑）、23（细纹理）、41（花叶）和83（NR下降最明显的负例）展示。原图已下载至上级nr_cuda_outputs/、cuda_outputs/，比较页在nr_recheck_visual_20261014/；原像素ROI与缩略概览分开标注。

[打开5组对比页](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_recheck_visual_20261014/index.html>) · [100张逐图变化CSV](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_metric_recheck_20261014/per_image_before_after.csv>)

| 原脚本完整复测版本 | test100 NR | val3 FR | S |
|---|---:|---:|---:|
| compiled6000 基线 | 2.253171080906 | 1.463928957780 | 3.717100038686 |
| conservative6250 | 2.266385043359 | 1.463716487090 | 3.730101530449 |

| test100均值 | 基线6000 | 最新6250 |
|---|---:|---:|
| CLIPIQA ↑ | 0.722625471 | 0.732439575 |
| MANIQA ↑ | 0.499498821 | 0.512805884 |
| MUSIQ ↑ | 43.187275162 | 42.354991055 |
| NIQE ↓ | 4.008259636 | 4.024103260 |

NR增加0.013213962，S增加0.013001492；改善主要来自MANIQA/CLIPIQA，MUSIQ、NIQE及FR有小幅回退。此次没有重新测速，沿用此前同卡G4校准的原工具估计：6000=54.8613，6250=55.5152；不作为正式榜分。

已下载5例的10张原4K PNG及原输入，全部SHA/尺寸匹配完整manifest，原文件未处理。总览为1/8缩略图，4组ROI为1:1原像素裁剪。6250相对6000肉眼变化较小；窗格仍有亮点、重复纹理，花瓣偏平滑、叶边偏硬。case83作为指标下降最明显的负例一同展示，避免只看改善样本。

[逐图变化汇总JSON](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_metric_recheck_20261014/per_image_change_summary.json>) · [可视化来源/ROI/SHA](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_recheck_visual_20261014/preview_manifest.json>)

[原脚本复测最终CPU审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_metric_recheck_20261014/audit.json>)
