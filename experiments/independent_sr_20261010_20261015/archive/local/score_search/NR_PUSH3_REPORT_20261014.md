# 第三轮 CUDA 评分优化（2026-10-14）

本轮已实际完成三条各750步的FP32 LoRA微调、九组小样本评测与三组完整评测。完整100测试+3验证最终选中 **C7250：重锚6500、加强NR**，NR **2.31312513**、FR **1.44656901**、S_proxy **3.75969413**；相对父6500，NR增加0.03481477、FR下降0.01487168、S净增0.01994309。原评分工具新鲜同卡配对预测 **55.55522 → 56.41263 分**，是经验估计，不是官方榜分。新模型512×512单卡p10 **46.82435ms**。

服务器为30416端口，实验根目录`/cache/independent_sr_cuda_20261013`。三条分支各使用一张A100（GPU0/1/2），均从同一sealed6500 FP32 LoRA、Adam500和四组RNG起步，各增加750步，目标global7250/Adam1250。每250步保存6750、7000、7250的推理权重和精确Adam/RNG续训状态，避免最终选中中间权重后无法精确续训。三分支不能相加称为同一模型新增2250步。

| 分支 | 冻结保真teacher | NR reward总系数 | 训练CLIPIQA系数 | MUSIQ系数 | 学习率 |
|---|---|---:|---:|---:|---:|
| A：延续旧目标 | exact FP32 6250 | 0.075 | 0.1 | 1.5 | 2e-7 |
| B：重锚当前最佳 | exact FP32 6500 | 0.075 | 0.1 | 1.5 | 2e-7 |
| C：加强NR | exact FP32 6500 | 0.15 | 0.5 | 1.5 | 2e-7 |

损失仍为`L1(teacher)+0.1*LPIPS(teacher)+0.3*spatial_gradient_L1 − reward_weight*(1.5*MUSIQ/100+MANIQA+clip_weight*CLIPIQA)`。A保持父训练目标连续；B仅更换teacher；C另增强NR奖励，属于探索性目标变化，不把其Adam继承误称为同一损失的连续优化。指标网络与teacher冻结，NIQE仅评测。正式评分仍为原工具固定公式，训练系数不改变正式评分系数。

初始CPU审计通过：514份LoRA和1028份Adam矩逐元素精确迁移，所有514份参数梯度有限且非零，MUSIQ/MANIQA/CLIPIQA对输入的真实梯度均非零。独立重放父Python RNG得到首case9/crop[3135,193]，三组完全一致；更新前的三个NR指标浮点值也完全相同。A初始fidelity=0.00808394；B/C均约2.25e-12，符合重锚到当前student后的预期。旧v1/v2/v3及父config/resume/source哈希保持封存。

训练代码：`/cache/independent_sr_cuda_20261013/code/cuda_training/train_cuda_nr_stage_v4.py`，SHA256 `5f2d40b41034db7f9ae5f6a5f64d90b77bff6bc849422e1fc03ba3eb1600f311`。配置同目录`config_stage4_A_continue6250.json`、`config_stage4_B_reanchor6500.json`、`config_stage4_C_nrstrong.json`。新代码有自己的显式parser和阶段循环，未覆写或动态替换旧训练入口。

训练目录分别为：

- `/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage4_s6500_teacher6250_cont750/`
- `/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage4_s6500_teacher6500_cont750/`
- `/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage4_s6500_teacher6500_nrstrong750/`

每个目录中的`step6750.pth`等是FP32推理LoRA；`resume_step6750.pt`等含同一步的FP32 LoRA、Adam和Python/NumPy/TorchCPU/CUDA RNG。保存先原子替换`resume_latest.pt`，再硬链接为不可变里程碑，后续保存替换新的latest inode；`checkpoint_step6750.json`在权重与恢复状态SHA核对后完成写入，作为可供评测的完成信号。里程碑审计读取immutable resume，不读取正在更新的latest。

实时状态：`/cache/independent_sr_cuda_20261013/results/status_cuda_nr_push3_20261014.json`；训练日志：`/cache/independent_sr_cuda_20261013/logs/push3_train_{A,B,C}.log`。训练初始及里程碑CPU证明在`/cache/independent_sr_cuda_20261013/audits/push3/training_stage4/`；本地副本位于[stage4审核目录](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/cuda_training/stage4>)。

评测使用原冻结`render_cuda_submission.py`、`score_cuda_stream.py`与原指标权重，候选只改变训练权重路径。固定test20+val3 pilot为23图/98标量，只称S_subset；正式test100+val3为103图/418标量。新目录为`nr_push3_pilot_evaluations/`与`nr_push3_full_evaluations/`。父6500旧pilot的23图、98指标、实际输入/GT/base/8份指标权重与源码/RNG已重新CPU核验，可在相同scope内作为对照；不能从父full抽20图当pilot。

本轮训练仍使用100测试输入作无GT的NR优化，验证GT未参与训练。这是针对这套评分任务的选型，不宣称独立测试泛化。最终结论以完整S和各分项为准，若NR上升但FR损失更大，则保留原最佳。

存储计划：旧两轮备份保持原样；新候选全部保留在空间充足的`/cache`。共享存储和本机空间有限，最终优先持久化新胜者的raw权重、对应里程碑的完整续训状态、103张正式原PNG及小证明，不把所有探索分支大文件重复复制到共享存储。本地下载小指标与少量原图案例。完成标记和独立归档回执见文末位置。



## 完整评测与选型

下面每行均为实际103张原PNG、418个标量的完整100测试+3验证，均通过独立CPU审计；S_proxy=NR(test100)+FR(val3)。使用相同原图、GT、原评分网络、源文件及随机状态协议。全部三候选完成后按最高S_proxy选中C7250，C7250相对C7000仅增加0.00089998，没有重复随机试验，不宣称统计显著性。

| 版本 | NR | FR | S_proxy | 相对6500变化 |
|---|---:|---:|---:|---:|
| push3_c7250_compiled_all | 2.31312513 | 1.44656901 | 3.75969413 | +0.01994309 |
| push3_c7000_compiled_all | 2.30915348 | 1.44964067 | 3.75879415 | +0.01904311 |
| push3_c6750_compiled_all | 2.30120160 | 1.45518788 | 3.75638947 | +0.01663843 |
| nr_stage3_anchor6250_m15_s6500_compiled_all | 2.27831036 | 1.46144068 | 3.73975104 | +0.00000000 |


小样本只用于先筛选分支，固定test20+val3，23图/98标量，以下只称S_subset。A/B未做完整103张评测；不能把pilot数值当作完整分数。

| 版本 | NR | FR | S_subset | 相对6500变化 |
|---|---:|---:|---:|---:|
| push3_c7250_compiled_pilot | 2.30585361 | 1.44656901 | 3.75242262 | +0.01466634 |
| push3_c7000_compiled_pilot | 2.30247818 | 1.44964067 | 3.75211886 | +0.01436258 |
| push3_c6750_compiled_pilot | 2.29515577 | 1.45518788 | 3.75034365 | +0.01258738 |
| push3_b7250_compiled_pilot | 2.29034084 | 1.45702660 | 3.74736744 | +0.00961116 |
| push3_b7000_compiled_pilot | 2.28710611 | 1.45708134 | 3.74418745 | +0.00643118 |
| push3_a7250_compiled_pilot | 2.28333058 | 1.45898048 | 3.74231106 | +0.00455478 |
| push3_b6750_compiled_pilot | 2.28365341 | 1.45842093 | 3.74207434 | +0.00431806 |
| push3_a7000_compiled_pilot | 2.28235278 | 1.45952882 | 3.74188160 | +0.00412533 |
| push3_a6750_compiled_pilot | 2.27952806 | 1.46111391 | 3.74064197 | +0.00288569 |
| nr_stage3_anchor6250_m15_s6500_compiled_pilot | 2.27631559 | 1.46144068 | 3.73775627 | +0.00000000 |


C7250完整测试均值：CLIPIQA 0.771804400、MANIQA 0.570102143、MUSIQ 39.666658401、NIQE 4.254479984。相对父6500，NR贡献分别为CLIPIQA +0.030101922、MANIQA +0.043942995、MUSIQ/100 -0.019426214、NIQE项 -0.019803934，净+0.034814769。FR中LPIPS和DISTS均恶化，不能称为全部指标改善或更真实细节。

正式NR固定为CLIPIQA+MANIQA+MUSIQ/100+max(0,(10−meanNIQE)/10)，FR为2−meanLPIPS−meanDISTS。正式MANIQA仍100 crops，训练20 crops；NR FP32，FR FP16 autocast。推理输入原尺寸4K FP16[-1,1]、内部BF16、输出FP32[-1,1]并round为uint8 PNG。HYPIR rank256、model_t200/coeff_t175、空提示、缓存、LoRA融合与原tile compile均不变。pilot与full的随机序列按各自完整case顺序推进，不能混用同编号PNG。

## 新鲜同卡测速及原工具预测

G4、父6500、新C7250分别在同一物理A100 GPU0顺序启动独立进程，各warm30/n200、seed231、512×512 CUDA FP16。UUID `GPU-2018f8c4-5c1f-600d-186a-eba0459e1b10`，PCI `00000000:9D:00.0`，每次开始前该卡无其他计算进程，200份event/wall原始样本已保存。编译/加载/warmup/hash不计入推理时间。

| 项目 | 本次实际值 |
|---|---:|
| G4参考p10 | 0.650240004ms |
| 父6500 p10 | 47.177521896ms |
| C7250 p10 | 46.824345398ms |
| 父6500本次工具预测 | 55.555223783 |
| C7250本次工具预测 | 56.412631062 |
| 固定本次父速度，仅质量变化 | 56.328523847 |

质量因子为1.013919484287，本次计时因子1.001493154963。约0.353ms计时差不证明算法稳定加速；不拿旧轮55.76799与本轮新参考值混比。原predict.py SHA256 `0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1`，未修改公式或评分网络。CPU join重新读取103原PNG/逐图指标、全部实际依赖与200计时样本，质量/速度绑定同一权重和推理配置。

完整原始记录见[同卡预测核验](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/calibration/push3_paired_prediction.json>)及同目录 `push3_{g4,parent6500,selected}_latency.json`、`push3_latency_launch.json`。服务器同位置在`/cache/independent_sr_cuda_20261013/calibration/`。

## 选中权重、精确续训与复现

- 原始权重：`/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage4_s6500_teacher6500_nrstrong750/step7250.pth`；SHA256 `fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e`。
- 精确续训状态：`/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage4_s6500_teacher6500_nrstrong750/resume_step7250.pt`；SHA256 `b8b709f284ce627113e392ede00e43bcd522c8dc23b63719e9fcbaa573c7256d`。
- 训练配置：`/cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage4_s6500_teacher6500_nrstrong750/config.json`，SHA256 `8bcf24e6f8965593f26d31aeedcc0c58fc62af185d443d8ff46d2602336b59f7`；local750/global7250/Adam1250。
- 实际推理模型目录：`/cache/independent_sr_cuda_20261013/submissions/hypir_push3_c7250_compiled/model_dir`。
- 正式103张输出：`/cache/independent_sr_cuda_20261013/nr_push3_full_evaluations/push3_c7250_compiled_all/render/{test,val}/`；逐图指标：`/cache/independent_sr_cuda_20261013/nr_push3_full_evaluations/push3_c7250_compiled_all/scores/measurements.jsonl`；汇总：同目录`result.json`。
- 正式选择：[nr_push3_selection.json](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push3_selection.json>)；选择文件SHA256 `dc41fefc4077b6b157e310065d5017dfc8c0c2fa791e943543abd97a9a91d6af`。
- 全量独立审核：`/cache/independent_sr_cuda_20261013/audits/push3/evaluations/push3_c7250_compiled_all/audit.json`，SHA256 `566943a05c2e64d457f74b8962caff4b84f043ded3a5c565bab6568b2c769838`；训练独立审核：`/cache/independent_sr_cuda_20261013/audits/push3/training_stage4/milestone750_cpu_audit.json`，SHA256 `da243b9fee82d389d84daef0876f0fdaa917b622df408b7fe0b9ee326cab1926`。
- 本地[完整公平表](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push3_20261014/evaluation_audits/fair_full_table.json>)与[全部评测小型证明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push3_20261014/evaluation_audits/>)；实际大权重留在服务器。

共享SD2基础权重在`/temp/zihaohe_stage/baselines/sd21b`；本轮不重复复制。原推理runner SHA256 `396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae`。训练与评测源均封存，续训应从上面的匹配immutable resume启动新run，不覆盖本轮结果。

## 可视化与负例

[最终4例对比图库](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push3_visuals_20261014/index.html>)包含case7、16、41、83的父6500/C7000/C7250共12张原4K PNG、原像素1:1同坐标ROI和1/8概览。所有原图逐一匹配最终full render_manifest，裁剪逐像素检查，未锐化或润色。肉眼差异较小，结构大致稳定，部分平滑、硬边及细纹问题依然存在，不能把NR上升直接解释为更真实的重建。

[验证GT负例诊断](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push3_val_visuals_20261014/index.html>)保留val1 GT、父6500、C7000和C7250。val1的LPIPS从0.491943升至0.507324、DISTS从0.210620升至0.221888，FR从1.297437降至1.270787。选取ROI依据新旧输出差异，不使用GT选区域；其局部L1可能微好而全图感知指标恶化，单个ROI不能解释全图全部误差。

主图库final audit SHA256 `3bb46db83d8413162c2ca5d1a94150d619ce24b83c912e0800883d4775a14d12`。小型图库源码/原图来源SHA/指标/ROI证明上传`/cache/independent_sr_cuda_20261013/audits/push3/visuals/`；本机原PNG留上述图库，不重复上传共享存储。

## 持久化范围和完成证据

本轮独立归档目的地：`/temp/independent_sr_cuda_20261013_artifacts/nr_push3_20261014`。只复制选中C7250的raw权重、匹配immutable resume、完整103原PNG、三分支CLI配置、v4源码、运行小日志和各项审计/评分/测速/报告。约6.3GB，避免重复全部探索权重和输出；未选中分支的大文件仍保留/cache。本机只下载小型证据和少量案例，不下载完整训练状态。

提交以该目录`DONE`内的manifest SHA和逐文件完整SHA读回为准。独立回执保存在`/cache/independent_sr_cuda_20261013/results/nr_push3_backup_receipt_20261014.json`，本地[归档回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push3_backup_receipt_20261014.json>)在提交核验后写入。报告本身在复制前冻结，回执单独生成，避免自引用manifest。实时状态在`/cache/independent_sr_cuda_20261013/results/status_cuda_nr_push3_20261014.json`，提交后的最终complete以该状态和回执为准。

旧`nr_microtune_20261014`与`nr_push2_20261014`两备份保持原字节，既有基础依赖由旧备份和共享SD2共同提供。新目录不是独立复制整套环境的快照。

额外的reduce-overhead推理探索只完成3次512输出/RNG逐bit相同；候选测速在预算截止时未完成，因此未采用，没有宣称加速或native4K等价。未来可研究全图上下文奖励与融合BF16的训练代理；本轮只保存安装源码分析，未把这些未实现方案写入已选模型。
