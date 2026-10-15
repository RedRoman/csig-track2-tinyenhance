# 第四轮 CUDA 短微调与完整复测（2026-10-14）

本轮完成三路各500步微调及额外一路250步短试、六个固定小样本评测和三组完整103张复测。最终选中 **push4_a7750_compiled_all**：NR **2.319733279**、FR **1.441723287**、S_proxy **3.761456566**。相对7250步父模型，NR +0.006608150、FR -0.004845719、S +0.001762432。

本次新鲜同卡原工具预估 **58.45892 → 58.69787 分**；固定本次父模型速度，仅质量变化为 **58.53038**。这些是原五点经验工具预测，不是官方榜分。512×512 A100单卡event p10为 **46.34194ms**；不得将其说成原4K全图延迟。

## 本轮训练的变化

端口30416，实验根目录 `/cache/independent_sr_cuda_20261013`。A/B/C三路从同一sealed C7250 FP32 LoRA、Adam1250和Python/NumPy/Torch CPU/CUDA四组随机状态起步，各训练500次，保存global7500/Adam1500和global7750/Adam1750；额外D从同一起点训练250次，保存global7500/Adam1500。四路是独立分叉，不是同一模型累计1750步。

| 分支 | MUSIQ训练系数 | 总NR奖励系数 | 其他参数 |
|---|---:|---:|---|
| A：原目标延续 | 1.5 | 0.15 | 共同参数见下 |
| B：提高MUSIQ | 2.5 | 0.15 | 共同参数见下 |
| C：降低总奖励 | 2.5 | 0.10 | 共同参数见下 |
| D：进一步降低奖励，250步 | 2.5 | 0.075 | 共同参数见下 |

共同参数：teacher6500精确FP32冻结权重、LR2e-7、CLIPIQA训练权重0.5、L1保真1、LPIPS保真0.1、空间梯度保真0.3；MANIQA训练20 crops。损失为 `L1+.1*LPIPS+.3*gradient_L1 − reward_weight*(musiq_weight*MUSIQ/100+MANIQA+.5*CLIPIQA)`。NIQE仅用于评测。A是相同损失的精确延续；B/C显式改变目标并继承Adam矩，不将其描述为原损失的连续优化。

之前最后250步的NR增益约77%被FR回退抵消，因此本轮比较奖励与保真的平衡。实际六组pilot表明，单独提高MUSIQ系数B在两个步数都降低了正式MUSIQ和综合S；C通过降低总奖励改善了B的保真损失。C7750首组完整S仅增加0.001158，仍很微弱，因此随后真实追加D250，只将总奖励改为0.075，并直接做同协议103图完整评测，未将其填入不存在的pilot结果。超参变化不保证对应正式指标单调上升。

训练源 `/cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr_stage_v5.py`，SHA256 `4b5bdf40aab5d2fd35c92f8fc714c675e62bd86228bc3fb79e48e896b70278f2`。四份实际CLI配置与说明位于 `/cache/independent_sr_cuda_20261013/code/cuda_training/push4/`。独立CPU审计覆盖初始迁移和实际里程碑：514份FP32权重有限且真实变化、原始权重与不可变resume逐值相等、1028份Adam矩正确、四组RNG可恢复；相同步数的case/crop和终点RNG对照一致；未初始化CUDA。首case93/crop[60,1399]与首步三个指标逐值一致。证据见[三路训练审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage5/>)、[D短试审计](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage5_D/>)和服务器同名 `audits/push4/training_stage5*` 目录。

仍使用这100张测试输入做无GT的NR优化，验证GT不参与训练。本报告描述这套评分任务上的选型，不声称独立测试集泛化。

## 原工具评测

正式NR固定为 `CLIPIQA+MANIQA+MUSIQ/100+max(0,(10−meanNIQE)/10)`；FR为 `2−meanLPIPS−meanDISTS`；S_proxy=NR(test100)+FR(val3)。正式MANIQA仍100 crops、NR FP32、FR FP16 autocast，评分代码、网络权重和公式未改。

推理仍为HYPIR rank256、model_t200/coeff_t175、空提示、BF16内部计算、缓存+LoRA融合+原tile compile。原尺寸输入FP16[-1,1]，输出FP32[-1,1]并round成uint8原尺寸PNG。原runner SHA256 `396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae`。各候选只改变checkpoint和label，source、输入、GT、原指标及随机策略由独立审计实际SHA验证。

下表均为完整103张/418标量独立审核PASS；在三组完整结果齐备后按最高S选型，并与父7250比较：

| 版本 | NR | FR | S_proxy | 相对7250变化 |
|---|---:|---:|---:|---:|
| push4_a7750_compiled_all | 2.31973328 | 1.44172329 | 3.76145657 | +0.00176243 |
| push4_c7750_compiled_all | 2.31701313 | 1.44383905 | 3.76085218 | +0.00115805 |
| push3_c7250_compiled_all | 2.31312513 | 1.44656901 | 3.75969413 | +0.00000000 |
| push4_d7500_compiled_all | 2.31108248 | 1.44778933 | 3.75887181 | -0.00082232 |


以下是固定test20+val3、23图/98标量小样本，只称S_subset；用于筛选，不作为完整S或工具榜分：

| 版本 | NR | FR | S_subset | 相对7250变化 |
|---|---:|---:|---:|---:|
| push4_c7750_compiled_pilot | 2.31203450 | 1.44383905 | 3.75587355 | +0.00345093 |
| push4_a7750_compiled_pilot | 2.31225586 | 1.44172329 | 3.75397915 | +0.00155653 |
| push4_c7500_compiled_pilot | 2.30835208 | 1.44473563 | 3.75308771 | +0.00066509 |
| push4_a7500_compiled_pilot | 2.30862629 | 1.44379685 | 3.75242314 | +0.00000053 |
| push3_c7250_compiled_pilot | 2.30585361 | 1.44656901 | 3.75242262 | +0.00000000 |
| push4_b7750_compiled_pilot | 2.31513666 | 1.43548314 | 3.75061980 | -0.00180282 |
| push4_b7500_compiled_pilot | 2.30989198 | 1.43959302 | 3.74948500 | -0.00293762 |


完整100张测试分项均值：

| 指标 | 父7250 | 本轮选中 |
|---|---:|---:|
| clipiqa | 0.771804400 | 0.776279178 |
| maniqa | 0.570102143 | 0.575256151 |
| musiq | 39.666658401 | 39.583499126 |
| niqe | 4.254479984 | 4.276370416 |

NR增量分解：CLIPIQA +0.004474778、MANIQA +0.005154008、MUSIQ/100 -0.000831593、NIQE项 -0.002189043，合计 +0.006608150；再加FR变化 -0.004845719 得到完整S变化。保留所有分项及负例，不将NR改善等同于每项指标或真实细节都改善。完整证据见[评测审核目录](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push4_20261014/evaluation_audits/>)。

## 本次同卡测速

G4、父7250、新选中模型分别用独立新进程，在同一物理GPU0顺序warm30/n200、seed231、512×512 FP16；每次开始前无其他计算进程。编译、加载、warmup和hash不计推理时间。

| 项目 | 本次实测/原工具计算 |
|---|---:|
| G4 event p10 | 0.781311989ms |
| 父7250 event p10 | 47.014706802ms |
| 新模型event p10 | 46.341939163ms |
| 父7250本次预估 | 58.458920267 |
| 新模型本次预估 | 58.697872019 |
| 固定父速度、仅质量变化 | 58.530378758 |

物理卡 `GPU-2018f8c4-5c1f-600d-186a-eba0459e1b10`、PCI `00000000:9D:00.0`。质量因子 1.001222371030；本次计时因子 1.002861646637，单次细小延迟差不证明算法稳定加速。只比较本次新鲜配对结果，不混用上轮56.41263及不同G4标定。

原predict.py SHA256 `0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1`，原benchmark SHA256 `3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995`。CPU join重新核验实际完整PNG、逐图指标、权重与推理配置以及三组各200份原始样本。见[配对预测核验](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/calibration/push4_paired_prediction.json>)和同目录 `push4_*_latency.json`、`push4_latency_launch.json`。

## 权重、续训和原图

- 选型：[nr_push4_selection.json](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push4_selection.json>)，SHA256 `b3148b21f466854f9fe8f81c7eca1348da5bf793141799573ff3be8b20e25c3f`。
- 原始LoRA：`/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth`，SHA256 `589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d`。
- 精确不可变resume：`/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/resume_step7750.pt`，SHA256 `e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b`；local500/global7750/Adam1750。
- 配置：`/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/config.json`；训练run `cuda_nr_stage5_s7250_teacher6500_continue500`。resume包含同一步514份FP32权重、Adam矩/步数和四组RNG，续训使用新run，不覆盖父模型。
- 实际推理模型：`/cache/independent_sr_cuda_20261013/submissions/hypir_push4_a7750_compiled/model_dir`。
- 完整103张原PNG：`/cache/independent_sr_cuda_20261013/nr_push4_full_evaluations/push4_a7750_compiled_all/render/{test,val}/`；逐图指标：`/cache/independent_sr_cuda_20261013/nr_push4_full_evaluations/push4_a7750_compiled_all/scores/measurements.jsonl`；正式汇总同目录 `result.json`。
- 训练独立审核：`/cache/independent_sr_cuda_20261013/audits/push4/training_stage5/milestone500_cpu_audit.json`，SHA256 `9d4e72d67a9c19f517c7928bc3311fbaef9cf1ef93bbd4a506872732dc86ae47`。
- 完整独立审核：`/cache/independent_sr_cuda_20261013/audits/push4/evaluations/push4_a7750_compiled_all/audit.json`，SHA256 `ae5b2234f5e110240507fbec515139d1b2a23bf8f664e7477e34c377e09c507b`。

[最新4例对比图库](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push4_visuals_20261014/index.html>)保留父7250及本轮完整候选在case7、16、41、83的原4K PNG、同坐标1:1 ROI和1/8概览。父图复用旧原PNG硬链接，新图必须匹配最终full manifest；裁剪逐像素核验，不锐化、不润色。视觉变化较细，不能据评分微增宣称明显结构改善。

[验证GT负例](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push4_val_visuals_20261014/index.html>)使用固定旧val1 ROI[2304,2688,2944,3200]，保留GT/父7250/候选原图与感知指标；未用GT重新选区域。单个局部裁剪不能解释整图全部FR变化。图库与诊断源码/来源SHA/最终检查记录在 `/cache/independent_sr_cuda_20261013/audits/push4/visuals/`。

## 持久化和精确恢复

新归档prefix `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014`。旧三轮归档保持原样。由于共享存储余量有限，本轮只持久化选中的完整immutable resume、原始raw ZIP的精确重建recipe/gaps、4张原尺寸案例以及新源码/配置/全部小型训练与评测证明。

**完整103张PNG保留/cache，未全部复制到共享存储；raw没有重复占用1GB，但可由resume和recipe逐字节重建成原文件。** 全部103张仍逐张核验SHA并保留指标与manifest。recipe不是精度压缩：514份FP32 tensor字节来自同一步resume，原ZIP其余字节原样保留，已实际重建完整raw并验证原SHA。

选中recipe目录 `/cache/independent_sr_cuda_20261013/audits/push4/raw_recovery/push4_a7750_compiled_all`，实际恢复证明 `/cache/independent_sr_cuda_20261013/audits/push4/raw_recovery/push4_a7750_compiled_all/probe_result.json`。共享归档mirror相同相对路径。恢复时用归档 `code/push4/checkpoint_archive_recipe.py restore --directory <recipe目录> --resume <immutable-resume> --output <新的raw输出路径>`，成功后raw完整字节SHA必须为 `589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d`。工具仅用Python标准库，拒绝覆盖已有输出。

复制助手只在全部源稳定、实际空间充足时流式写入新prefix，再完整SHA读回，最后写manifest和DONE。提交以 `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/DONE` 的manifest SHA为准。提交后独立回执在 `/cache/independent_sr_cuda_20261013/results/nr_push4_backup_receipt_20261014.json`，本地[归档回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push4_backup_receipt_20261014.json>)在独立核验后写入；报告复制前冻结，回执单独保存避免自引用。

共享基础SD2仍在 `/temp/zihaohe_stage/baselines/sd21b`，原输入、GT、指标及旧代码依赖由旧三轮prefix和共享SD2共同提供，本轮不是整套环境的独立副本。实时状态 `/cache/independent_sr_cuda_20261013/results/status_cuda_nr_push4_20261014.json` 在提交后更新complete，归档内状态是复制前的选中快照。
