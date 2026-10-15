## 2026-10-15 当前采用HYPIR A7750画质基线

2026-10-15视觉反馈更新：**当前首选为HYPIR A7750（previous model output）**。用户认为新学生已有伪影，128.43分学生保留为评分/速度对照。后续以A7750画质为基线，在文字、直线与细边界不出现明显新增伪影的前提下提分；本次没有启动训练。[当前模型指针](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/current_model_preference.json>) · [选择说明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/VISUAL_SELECTION_20261015.md>) · [视觉对比](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push5_visuals_20261014/index.html>)。下方为原实验历史。

## 2026-10-14 第五轮：轻量学生超过70分目标

2026-10-14第五轮轻量学生：已达到原工具70分目标，新学生预估 **128.42938分**。从用户工具包训练好G4初始化，向A7750原生test100伪标签独立蒸馏250步，新Adam/RNG；完整103图/418标量 NR **2.47546636**、FR **1.36650445**、S **3.84197081**，同A100单卡512 FP16 p10 **0.78643203ms**。本次配对HYPIR58.15040、随包G4参考复现152.66092均单列；这些是原工具经验预测，不是官方榜分。新学生部分波纹减弱，但细结构仍弱于HYPIR，不宣称全面画质提升。约11.8MB提交包和完整续训状态已保存本地；共享归档以新nr_push5_20261014/DONE及[独立回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push5_backup_receipt_20261014.json>)为准。[提交包](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push5_20261014/pix_push5_student250_model_dir.zip>) · [完整报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_PUSH5_REPORT_20261014.md>) · [4例原图对比](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push5_visuals_20261014/index.html>) · [权重及评分路径](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push5_selection.json>)。30416端口，根目录/cache/independent_sr_cuda_20261013。以下保留历史记录。

## 2026-10-14 第四轮CUDA短微调已完成

2026-10-14第四轮CUDA短微调：完成三路各500步及额外一路250步训练、6组pilot与3组完整100测试+3验证评分。最终选中 **A7750／原目标继续500步**，NR **2.31973328**、FR **1.44172329**、S_proxy **3.76145657**；相对父7250，NR+0.00660815、FR-0.00484572、S+0.00176243。本次同A100单卡新鲜配对的原工具预估 **58.45892→58.69787分**；固定本次父速度仅质量变化为**58.53038分**，均为经验预测。512×512 event p10 **46.34194ms**。本轮增益较小，约73%的NR收益被FR回退抵消；肉眼主要是细微纹理差异。全部103原PNG保留/cache；共享归档仅保存精确resume、可逐字节恢复raw的recipe、4例原图和小型证据，完成状态以新prefix `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/DONE` 及[独立归档回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push4_backup_receipt_20261014.json>)为准。[完整中文报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_PUSH4_REPORT_20261014.md>) · [最新4例对比](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push4_visuals_20261014/index.html>) · [GT负例](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push4_val_visuals_20261014/index.html>) · [权重与续训位置](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push4_selection.json>)。30416端口，实验根目录 `/cache/independent_sr_cuda_20261013`。以下保留历史记录。

## 2026-10-14 第三轮CUDA NR优化已完成

2026-10-14第三轮CUDA NR优化：三条分支从同一6500父模型分别新增750步，完成9组小样本及3组完整100测试+3验证评测。最佳为 **C7250／重锚6500／加强NR**，NR **2.31312513**、FR **1.44656901**、S_proxy **3.75969413**；相对上一轮6500，NR+0.03481477、FR−0.01487168、S净+0.01994309。新鲜同A100单卡配对测速p10 **46.82435ms**，原工具预估 **55.55522→56.41263分**；固定父模型速度仅质量变化为56.32852分，均为经验预测。MANIQA/CLIPIQA上升，MUSIQ/NIQE与FR下降；肉眼变化较小，不能解释为全部细节更真实。选中权重、匹配完整Adam/RNG续训状态和103原PNG共400文件（6,294,606,619 bytes）已由复制助手逐文件SHA回读后提交至共享存储 `/temp/independent_sr_cuda_20261013_artifacts/nr_push3_20261014/DONE`；独立最终核验以[归档回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push3_backup_receipt_20261014.json>)为准。[完整中文报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_PUSH3_REPORT_20261014.md>) · [4例原图和局部比较](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push3_visuals_20261014/index.html>) · [验证GT负例](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push3_val_visuals_20261014/index.html>) · [选型与权重路径](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push3_selection.json>)。服务器仍为30416端口，实验根目录`/cache/independent_sr_cuda_20261013`。以下保留历史记录。

## 2026-10-14 第二轮CUDA NR优化已完成

2026-10-14第二轮CUDA NR优化：四条分支各新增250步，完整评测三组新模型（每组100测试+3验证）。最佳为**父6250保真锚点/MUSIQ训练权重1.5/global6500**，完整NR **2.27831036**、FR **1.46144068**、S **3.73975104**，相对父6250的NR增加0.01192532、S增加0.00964951。单张A100 512×512 p10 **46.35085ms**；原工具本次同卡校准预测 **55.76799分**，固定父延迟仅质量变化的预测55.69676（父55.32547），不是官方榜分。四终点/优化器/RNG恢复已备份，最终评分归档以新共享目录`/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014/DONE`为准。[完整中文报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_PUSH2_REPORT_20261014.md>) · [新三版本可视化](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push2_visuals_20261014/index.html>) · [选型和权重位置](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push2_selection.json>)。以下保留历史记录。

## 2026-10-14 最新完成：30416 CUDA NR微调250步

从raw6000/coeff175起步，新增保守训练250步至global6250。100张测试+3张验证完整评分：NR **2.26638504**（基线2.25317108）、FR **1.46371649**、S **3.73010153**（基线3.71710004）。单张A100 512×512 p10 **45.78048ms**，原工具同卡校准预测 **55.5152**（基线54.8613），属于工具估计。完整评分、真实FP32更新与AdamW250/RNG恢复均通过审计；权重、恢复状态及输出已备份共享存储。详见[本轮中文说明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_FINETUNE_REPORT_20261014.md>)。下面保留历史阶段。

## 2026-10-13 最新完成：CUDA完整评分与单卡提速

训练已保存至10000步。本轮已完成参数筛选及三组CUDA完整复评，最佳预测版本为6000步/coeff175/缓存+LoRA融合+tile编译。100测试+3验证的代理S=3.71710004，单卡A100 512×512 p10=46.43932ms，用户工具预测54.8613分；原8000版本同口径预测44.1095分。工具预测不是官方分数。309张CUDA原PNG及全部结果已SHA校验并持久化；完整说明和路径见[实验说明](../csig_scoring_tool_review_20261013/score_search/README.md)。下面保留阶段历史。

## 2026-10-13 22:13 最新：完整评分与临时CUDA单卡校准

本轮训练权重已保存至10000步；本次没有启动新训练，而是使用已有6000/8000/10000步断点评分、筛选参数和优化推理速度。旧文档中的7000/8000等待器记录为历史阶段。

- 30207完成4组完整评分（测试100张+验证3张）：6000/coeff175 **3.71705244**；6000/coeff200 **3.68859777**；8000/coeff200 **3.66037796**；10000/coeff200 **3.66035744**。均为同NPU协议代理S，不是官方榜分。
- 当前最佳为6000步、model_t200、coeff_t175，比8000基线提升0.05667448；9组pilot用于筛选，不能作为完整分。全部13组输出和指标已校验并持久化，共享目录 `/temp/independent_sr_score_search_20261013_artifacts`。
- 30416是4张A100-SXM4-80GB的CUDA环境。单卡GPU0、512×512、warm30+n200 event p10：G4参考0.65024毫秒，8000基线113.29321毫秒，6000/175原版113.69799毫秒；缓存105.32751毫秒（输出/RNG检查完全一致）；缓存+融合85.14150毫秒；缓存+融合+编译46.43932毫秒（独立数值/质量版本）。
- CUDA原工具提交接口的FP16输入、round uint8无损PNG、原尺寸103图完整复评正在运行，输出位于 `/cache/independent_sr_cuda_20261013/evaluations/`。不得将旧NPU FP32/floor质量直接当作CUDA正式预测。
- 说明、结果和复现协议见 [实验说明](../csig_scoring_tool_review_20261013/score_search/README.md)，实时本地CUDA状态见 [status_cuda_search.json](../csig_scoring_tool_review_20261013/score_search/results/status_cuda_search.json)。

## 2026-10-13 最新：7000步实际审核完成，已排队纠偏续训至8000步

- 已实际查看全部24张中心/最大变化面板和3张固定位置纹理面板。case23玻璃规则点阵较6000减轻，但仍有残留；case16墙面、植物过度生成与标志畸变未解决，未批准为无伪影发布版。
- 固定12图：NIQE 3.603905，MUSIQ 46.083029，MANIQA 0.452881，CLIPIQA 0.716632，NR 2.269953。相对6000 NR略降，MUSIQ回升；仅本地代理指标，不是官方分数。
- 审核记录 code/ascend_review_step7000.json；下段维持lr=2e-7、reward_weight=0.075、gradient_weight=0.3，8000步后再次实际审核。
- 等待器PID2411563，日志logs/continue_7000_8000.log；8000评分等待器PID2411564。初次检查8张NPU均有进程；随后NPU3空闲，已启动续训，实测推进至7023步。
- 10000步本地备份旧等待器12小时超时，已重启PID32552，日志本地backup_key10000_retry.log。1000/5000已校验备份不变，10000尚未产生。

# 昇腾部署与恢复记录（2026-10-12）

## 2026-10-13 更新：6000步NR升高，但发现局部规则纹理，继续纠偏到7000步

- 6000步24张标准放大图全部实际查看，另补看case16/23/24同位置INPUT/原始HYPIR/5000/6000四列图。
- case23玻璃区域出现比5000和原始模型更明显的规则点阵/蜂窝状纹理。原始模型也有不规则生成颗粒，不能说完全由微调从零产生；但增强现象支持纠偏。case16墙面和case24天空颗粒也更强。
- 6000步仅作为有局部纹理风险的评分对照保留，未批准发布为“无伪影”版本。
- 固定12图指标：NIQE3.620247，MUSIQ43.550780，MANIQA0.476106，CLIPIQA0.735517，代理NR2.285106。高于5000步2.215626及1000步2.266707，但局部观感代价可见。
- 纠偏参数6000→7000：`lr=2e-7, reward_weight=0.075, gradient_weight=0.3`（此前分别5e-7、0.1、0.1）。原始冻结锚点、像素/LPIPS项不变。
- `review_step6000.json` 的批准范围仅为纠偏续训；`release_approved=false`，禁止原配方无条件继续。旧脚本保存在 `code/train_hypir_ascend_v1_pre6000.py` 和对应旧runner，旧结果不覆盖。
- 续训等待器PID2122896，日志 `logs/continue_6000_7000.log`；7000步评分等待器PID2122897。
- 新代码加入可配置梯度锚点权重，配置记录实际loss公式和脚本哈希；远端语法检查通过。7000步必须复核case23同一位置纹理，再决定继续。

## 2026-10-13 早先记录：5000 步审核及评分完成，调整奖励继续6000步

- 5000 步 12 张原尺寸预览和 24 张放大面板已生成并全部实际查看。未观察到明确新增周期纹或结构崩坏；既有文字畸变、花瓣蜡感、植物过度生成仍未解决。
- 同口径12图：NIQE3.548759，MUSIQ45.865191，MANIQA0.423422，CLIPIQA0.688428，代理NR2.215626。相对4000步2.216513基本持平，低于1000步2.266707。
- 评分结果 `results/native_subset_0_1000_4000_5000.json`。仅固定12图，不能代替官方分数、全100图质量或真实保真评估。
- 依据指标停滞，下一段5000→6000恢复最初已测试的 `reward_weight=0.10`，学习率保持 `5e-7`，保真项不变。审核记录 `review_step5000.json` 包含改动依据；不修改旧实验结果。
- 续训等待器PID2003442；日志 `logs/continue_5000_6000.log`。6000步审核后再放行。
- 6000步同口径评分等待器PID2003443，日志 `logs/native_subset_6000.log`；必须等完整预览后执行。
- 5000步关键权重已成功备份本地并完成远端/本地SHA256校验：`edab44c9d3e4df69f16b79e323faa473f560c52f23d1bcb42353a667b06e72bd`，1,038,080,515字节。配置、审核记录与manifest均保存在 `D:\competitions\KEY_CHECKPOINTS_ASCEND_20261012`。

## 2026-10-13 早先记录：4000 步审核完成，继续 5000 步；原尺寸评分已完成

- 4000 步的 24 张放大对照图已全部实际查看：与 3000 步视觉变化较小，没有观察到明确新增伪影；既有标志形变和植物过度生成未解决。
- `review_step4000.json` 仅批准受控续训，维持 `lr=5e-7, reward_weight=0.05`。4000→5000 已实际运行；等待器 PID 1875242；日志 `logs/continue_4000_5000.log`。
- 原尺寸固定 12 图评分已成功完成：`results/native_subset_0_1000_4000.json`，逐图分项在各 `review_previews/ascend_balanced_sSTEP/metrics_native_ascend_v1.jsonl`。
- 同一 JPEG100、seed1234、FP32、MANIQA100 crops 协议，指标如下。仅代表这 12 图、当前昇腾实现，不能当作官方分数或完整100图结果；跨设备数值一致性未校验。

| 步数 | NIQE↓ | MUSIQ↑ | MANIQA↑ | CLIPIQA↑ | 代理NR↑ |
|---|---:|---:|---:|---:|---:|
| 0 | 3.575405 | 44.384470 | 0.391767 | 0.655464 | 2.133535 |
| 1000 | 3.742595 | 42.842187 | 0.476509 | 0.736036 | 2.266707 |
| 4000 | 3.569192 | 45.840739 | 0.425100 | 0.689925 | 2.216513 |

- 4000 步代理NR相对原始 +0.082978，但相对1000下降0.050194。MUSIQ/NIQE更好，MANIQA/CLIPIQA回落。没有证据支持“步数越多越好”；5000步审核与同口径评分后再调整配方。
- NR按逐图 `CLIPIQA+MANIQA+MUSIQ/100+max(0,(10-NIQE)/10)` 后求均值。训练日志crop_NR采用CLIP权重0.1，不能直接与此比较。

## 2026-10-13 早先记录：3000 步审核完成，继续 4000 步

- 3000 步断点已保存；本段训练耗时 1329.46 秒，预览/对照生成完成。24 张放大面板已全部实际查看。
- 对照 2000 步：天空/墙面颗粒略少，主要轮廓基本稳定；未见明确新增周期纹。既有标志畸变、花瓣蜡感、枝叶过度生成仍在，尚未证实质量/分数提升。
- 审核记录 `adaptation/runs/ascend_balanced/review_step3000.json`；维持 `lr=5e-7, reward_weight=0.05`。
- 3000→4000 已实际推进超过 3100 步，等待器 PID 354439；日志 `logs/continue_3000_4000.log`。4000 步后必须停下实际审核。
- 审核图本地 `D:\competitions\round2_20261010\ascend_review_3000`，服务器 `review_previews/review_3000`。
- 通用续训入口 `code/queue_reviewed_segment.py STEP` 仅接受已实际完成并批准的对应审核记录；不得批量预填批准。

## 2026-10-13 早先记录：2000 步审核完成，继续 3000 步

- 2000 步训练完成，训练日志耗时 1337.03 秒；12 张全分辨率预览、24 张中心/变化最大区域的放大对照图均已生成并实际查看。
- 对照 1000 步，未在已查看区域发现明确新增周期纹或结构崩坏；但文字标志形变、花瓣蜡感、植物过度生成仍存在，不属于质量发布批准，也未证实涨分。
- 原始 HYPIR 与 1000 步的另外 24 张对照图已实际复核；部分问题原始模型已有。补充记录 `code/ascend_baseline_supplement_1000.json`。
- 审核记录 `adaptation/runs/ascend_balanced/review_step2000.json`；下一段维持 `lr=5e-7, reward_weight=0.05`。
- 2000→3000 已实际开始，使用 NPU 0；等待器 PID 238675，训练 PID 239171；日志 `logs/continue_2000_3000.log`。3000 步后停下审核，不自动放行。
- 总体阶段状态 `results/status_ascend_train.json`；实时训练步数请读取 `results/status_ascend_balanced.json` 或 `train.jsonl`，阶段状态不会逐步更新。
- 本地 1000 步权重备份已通过 SHA256；5000/10000 备份观察器仍在等待对应断点。

## 2026-10-13 早先记录（1000 步）

- 已完成 1000 步，训练日志计时 1278.79 秒；12 张全分辨率预览已生成。
- 已实际查看 12 案例 × 中心/最大变化区域，共 24 张 2 倍放大图。结论详见 `ascend_review_step1000.json`。
- 文字和硬边清晰，但 case25 标志形变、case17/41/44 花叶过度生成风险明显。原始 HYPIR 的同设置对照因显存不足失败，不能归因于微调或宣称优于原始模型。
- 仅批准受控续训实验：1000→2000 步使用 lr=5e-7，reward_weight=0.05；不代表发布质量通过。
- 32193 所有 8 卡现被其他任务占用；其余可连接的授权昇腾端口也无空卡。没有终止其他任务。
- 等待进程 PID 4179102 每 30 秒检查空卡，空闲后运行 `run_ascend_segment.py 1000`，2000 步后生成对照并停止审核，不自动批准下一段。
- 日志 `logs/continue_1000_2000.log`；状态 `results/status_ascend_train.json`。
- 本地 1000 步权重已成功备份，大小 1,038,080,515 字节，SHA256 `34f6f3619f44184c08be10c80b973fc61512cd0fe45fd7e0336956fc6d3f3fcf`。
- 本地 5000/10000 步备份观察器已启动，最长等待 12 小时；产生对应断点后才下载。

## 当前事实

- 原 A100 31904 已关闭；本轮无法读取旧断点，使用已校验的公开 HYPIR 初始化，不属于旧 1000 步训练的续训。
- 昇腾端口 32193，独立目录 `/cache/independent_sr_ascend_20261012`，拟使用 NPU 0。
- 模型、100 张输入、四项可微评分网络已部署；CPU 严格加载评分网络通过。
- 系统 CANN 缺少完整 910B 算子。已在实验自己的 `runtime/opp` 补入完整 910B 内核以及匹配的 OPP 配置与实现，未改系统目录。
- 独立运行库还需要 `runtime/hostlibs/data` 与 `runtime/share/info/runtime/version.info`；版本文件缺失曾导致设备能力查询失败，现已补齐。
- **BF16 512×512 矩阵乘法前向和反向已实际通过**，输出有限且梯度非零。
- 卷积和 BF16 注意力的反向传播也已通过，`runtime_ready` 已写入。
- 首次 HYPIR 启动加载模型成功，但调度器覆盖了 CANN Python 路径，卷积编译时无法导入 `tbe`；已修复路径保留逻辑并重启（调度器 PID 1472434，训练 PID 1472446）。
- **实际训练已成功完成至少 2 步参数更新**。第一步耗时 125.07 秒（含首次编译及冻结锚点加载），第二步增量约 1.16 秒；这只是初步耗时，不能据此保证全程速度。
- 首步可微梯度检查：MUSIQ 0.331058、MANIQA 0.209434、CLIPIQA 2.338521；LoRA 梯度范数 0.550638，有限且非零。初始保真梯度为 0 是因为学生与冻结锚点同权重同 latent，第二步保真损失已非零。
- 质量与涨分尚未经本轮检查；1000 步仍需实际视觉审核后继续。

## 操作入口

- `ascend_apply_opp_metadata.sh`：安全检查后导入配套 OPP，矩阵反向验证。
- `ascend_model_ops_probe.sh`：卷积与注意力反向验证；通过后写 `runtime_ready`。
- `start_ascend_train.sh`：启动独立训练调度器，首段 0–1000 步。
- 服务端状态：`results/status_ascend_train.json`；训练日志：`logs/train.log`。
- 每 1000 步必须实际查看图像后才可继续，不自动批准视觉审核。

## 本地备份

用户授权仅备份关键断点，工具 `backup_ascend_key_checkpoint.py` 支持 1000、5000、10000 三个里程碑。
本地目录 `D:\competitions\KEY_CHECKPOINTS_ASCEND_20261012`，校验远端与本地 SHA256。
未产生训练权重时不应宣称已有备份。
