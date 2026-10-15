# 第二轮CUDA NR优化（2026-10-14）

本轮继续使用30416端口，基于上一轮完整评分最佳的6250断点。旧6250完整NR=2.266385043359、FR=1.463716487090、S=3.730101530449。四组新增微调已完成，其中三组完成100测试+3验证的独立完整评分。当前最佳是parent-anchor M1.5的6500终点：NR=2.278310360288，FR=1.461440682411，S=3.739751042699。相对父6250的NR增加0.011925316929、S增加0.009649512250。此前55.5152分使用历史测速，本轮另做同次单卡G4与父/候选测速，最终预测不能混用两次校准。

四条训练分支均从同一sealed6250 FP32 LoRA、Adam250及四套RNG起步，各新增250步，终点均为本分支global6500/Adam500。它们是四个分叉，不能相加说成同一模型连续训练1000步。

1. stage2 continuous：延续原6000保真锚点，训练损失不变。
2. stage2 PNG-STE：同起点/锚点，仅NR前向使用正式float32 Runner/PNG量化往返，跨round梯度为恒等STE估计。
3. stage3 parent-anchor M1：保真锚点改为父6250的exact FP32权重，MUSIQ训练权重1。
4. stage3 parent-anchor M1.5：同6250锚点，MUSIQ训练权重1.5。

MUSIQ权重只改变训练loss，正式评分仍使用用户原工具固定公式；指标网络和权重始终冻结。训练MANIQA为20crop，正式为100crop；NIQE仅评测。native4K PNG输出、FP16输入/内部BF16、round uint8及原指标精度/RNG协议保持原记录。

训练初始FP32恢复、514 LoRA、Adam步数/状态、全部case/crop与最终四RNG均有CPU审核；两个stage3 teacher也精确恢复6250 FP32并冻结，首步fidelity为1.139e-14。旧源码、父config/resume SHA未变。训练入口为新v2/v3文件，旧run和原始代码保留。

| fixed20+val3 pilot | test20 NR | val3 FR | S_subset |
|---|---:|---:|---:|
| fresh6250 control | 2.266482978 | 1.463716487 | 3.730199465 |
| 6250 coeff160 | 2.247007562 | 1.470993261 | 3.718000823 |
| 6250 coeff185 | 2.270527323 | 1.456893822 | 3.727421144 |
| continuous6350 | 2.267105437 | 1.462449471 | 3.729554909 |
| continuous6500 | 2.268350612 | 1.462408304 | 3.730758916 |
| PNG-STE6500 | 2.268326264 | 1.462232073 | 3.730558337 |
| parent-anchor M1 6500 | 2.274525302 | 1.462559005 | 3.737084306 |
| parent-anchor M1.5 6500 | 2.276315593 | 1.461440682 | 3.737756275 |

八组pilot已全部完成并通过CPU逐文件/逐值审核。小样本仅为S_subset，不能冒充完整S。fresh6250对照的23PNG字节/像素、4组推理RNG和98项指标与旧6250 pilot全部精确一致。

| 完整100测试+3验证 | NR | FR | S_proxy |
|---|---:|---:|---:|
| 父6250 | 2.266385043359 | 1.463716487090 | 3.730101530449 |
| continuous6500 | 2.268540837285 | 1.462408304214 | 3.730949141500 |
| parent-anchor M1 6500 | 2.276198114365 | 1.462559004625 | 3.738757118990 |
| **parent-anchor M1.5 6500** | **2.278310360288** | **1.461440682411** | **3.739751042699** |

continuous6500的完整NR增加0.002155793926、FR降低0.001308182875、S净增加0.000847611051；418项指标和103张输出SHA已审核。两组parent-anchor也已完成，并通过独立103图/418项CPU审计；M1.5在完整NR和完整S上均领先，选为本轮最佳。每组原始输入103项、GT3项、PNG103项、指标权重8份、SD2底模13份及checkpoint/源码SHA均通过。与父6250的图像顺序和渲染/评分最终四RNG哈希相同。

同次单卡A100测速（512×512、FP16输入、warm30+n200、CUDA event p10，G4参考0.65126401186ms）与未修改的原`predict.py`联合结果：

| 模型 | p10/ms | 同卡校准工具预测 | 固定父6250延迟，仅质量变化的预测 |
|---|---:|---:|---:|
| 父6250 | 46.650162888 | 55.325474713 | 55.325474713 |
| M1 6500 | 46.588621140 | 55.672992927 | 55.658402973 |
| **M1.5 6500** | **46.350848389** | **55.767986476** | **55.696761230** |

三次微调使用同一种推理算法，约0.3ms的单次测速差不证明稳定提速。M1.5的S改善单独带来工具预测乘数1.00671095；固定父延迟时约+0.37129分，同次实测延迟代入约+0.44251分。此前父6250的55.5152使用另一轮45.78048ms测速，不宜与本次结果直接相减归因为训练收益。工具预测是原作者对五个轻量像素模型的经验拟合外推，不是官方榜分，也不把历史拟合残差当作HYPIR置信区间。完整联合证据见[paired_full_prediction.json](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push2_full_evaluations/audits/paired_full_prediction.json>)。

完整分项的实际变化如下；提高训练MUSIQ系数没有使正式MUSIQ同步上升：

| 测试100图均值 | 父6250 | 最佳M1.5 6500 | 对NR增量贡献 |
|---|---:|---:|---:|
| CLIPIQA（高为好） | 0.732439575 | 0.741702479 | +0.009262903 |
| MANIQA（高为好） | 0.512805884 | 0.526159148 | +0.013353264 |
| MUSIQ（高为好） | 42.354991055 | 41.609279842 | −0.007457112 |
| NIQE（低为好） | 4.024103260 | 4.056440646 | −0.003233739 |

验证FR从1.463716487降至1.461440682（−0.002275805），所以NR的全部增量不会等量变成S增量。三个新full结果不能只挑单项最优混成一个不存在的模型。

完整NR公式：CLIPIQA + MANIQA + MUSIQ/100 + max(0,(10-NIQE)/10)，各项先取100图均值再代入；FR=2−LPIPS−DISTS，来自3组验证参考。S=NR+FR。训练reward不是正式分，工具预测也不是官方榜分。

测试100图已参与无GT随机crop训练，不用于宣称独立泛化；验证GT未用于训练。本轮测试100图参与了同一数据域的直接NR优化，结果用于这套评分任务选型。

共享新备份：`/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014`。四组6500终点权重与各自完整Adam/RNG恢复状态已在checkpoint阶段逐文件SHA保存（101记录16,611,598,391字节；该阶段manifest SHA256为`ab0951571dbcba53c77a2433f2c3c2f6b0ef17aa4ab3a0c98291848daee643db`）。最终归档在此提交基础上追加三组full、八组pilot的493张原PNG及评分、代码、日志、审核证明；以目录根部`DONE`的`scope=final_all_pilots_and_formal_full_artifacts`、`finalize_pending=false`和匹配的`manifest_sha256`为最终完成依据。最后提交信息单独保存在[本地归档回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push2_backup_receipt_20261014.json>)，避免报告自引用自身manifest的SHA。新/变化文件逐一SHA回读，已提交大文件保留原记录并校验存在/尺寸，提交链保留历史。

父资产引用`nr_microtune_20261014`（标记为`DONE.json`），不重复复制SD2/env。最终备份保留四组终点和可续训状态，中间6350等权重不在本次选择范围；已评分中间输出与逐图指标保留。两种完成标记名不可混淆。

服务器根目录为`/cache/independent_sr_cuda_20261013`。训练代码位于`code/cuda_training/train_cuda_nr_stage_v2.py`与`train_cuda_nr_stage_v3.py`；每个`adaptation/runs/cuda_nr_stage*/`包含`config.json`、`train.jsonl`、`step6500.pth`、`resume_latest.pt`及初始化/梯度证明。权重用于推理，续训必须用对应的`resume_latest.pt`恢复FP32 LoRA、Adam500和四套RNG，并为新阶段建立新配置/目录，不能仅加载推理权重或覆盖冻结旧run。

| `adaptation/runs/`下目录 | `step6500.pth` SHA256 |
|---|---|
| `cuda_nr_stage2_s6250_cont250` | `dc7bbf712e3039302a52951075409a23771b8d832024d1800bbe3398c41d4afa` |
| `cuda_nr_stage2_s6250_roundste250` | `095f93f17d998d708639f2a3b3641052c576b3036efcbc881eb2e88360e18070` |
| `cuda_nr_stage3_s6250_anchor6250_m1_250` | `b05e784a08b7082a4ecae63a814d71834068ceb5e9baf26eded3836b29bcc0b8` |
| `cuda_nr_stage3_s6250_anchor6250_m15_250` | `cfcc3013fa87aa721f346d6cc285e70b6d6ff2c88315ce9f1465b07ffe020e68` |

对应提交Runner位于`submissions/hypir_cuda_nr_stage3_anchor6250_m1_s6500_compiled/model_dir`和`hypir_cuda_nr_stage3_anchor6250_m15_s6500_compiled/model_dir`。依赖底模`/temp/zihaohe_stage/baselines/sd21b`，本轮不复制env或SD2大文件；原始工具、指标权重、输入与验证GT在上一轮`nr_microtune_20261014`备份。恢复应联合两轮manifest及底模provenance检查，不能把checkpoint备份当作包含全部运行环境的容器镜像。

正式输出与逐图指标：`nr_push2_full_evaluations/<label>/{render,scores}`；pilot为独立的`nr_push2_pilot_evaluations/<label>/`。实时状态为`results/status_cuda_nr_push2_20261014.json`，运行日志为`logs/push2*`。本地CPU审核在`cuda_training/{stage2,stage3}/`、`nr_push2_pilot_evaluations/audits/`和`nr_push2_full_evaluations/audits/`，已上传服务器`audits/push2/`以便最终归档。

新[三版本可视化页](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push2_visuals_20261014/index.html>)比较父6250、M1 6500和M1.5 6500的正式full原图，保留1:1 ROI和单独标注的缩略总览。抽查中三版差异很小，窗格细碎亮点/重复边纹、花瓣平滑和叶缘偏硬仍存在，不能因指标提升便声称细节真实或肉眼明显更好。
