# 独立 HYPIR 参数评测 worker

2026-10-15视觉反馈更新：**当前首选为HYPIR A7750（previous model output）**。用户认为新学生已有伪影，128.43分学生保留为评分/速度对照。后续以A7750画质为基线，在文字、直线与细边界不出现明显新增伪影的前提下提分；本次没有启动训练。[当前模型指针](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/current_model_preference.json>) · [选择说明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/VISUAL_SELECTION_20261015.md>) · [视觉对比](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push5_visuals_20261014/index.html>)。下方为原实验历史。

2026-10-14第五轮轻量学生：已达到原工具70分目标，新学生预估 **128.42938分**。从用户工具包训练好G4初始化，向A7750原生test100伪标签独立蒸馏250步，新Adam/RNG；完整103图/418标量 NR **2.47546636**、FR **1.36650445**、S **3.84197081**，同A100单卡512 FP16 p10 **0.78643203ms**。本次配对HYPIR58.15040、随包G4参考复现152.66092均单列；这些是原工具经验预测，不是官方榜分。新学生部分波纹减弱，但细结构仍弱于HYPIR，不宣称全面画质提升。约11.8MB提交包和完整续训状态已保存本地；共享归档以新nr_push5_20261014/DONE及[独立回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push5_backup_receipt_20261014.json>)为准。[提交包](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push5_20261014/pix_push5_student250_model_dir.zip>) · [完整报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_PUSH5_REPORT_20261014.md>) · [4例原图对比](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push5_visuals_20261014/index.html>) · [权重及评分路径](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push5_selection.json>)。30416端口，根目录/cache/independent_sr_cuda_20261013。以下保留历史记录。

2026-10-14第四轮CUDA短微调：完成三路各500步及额外一路250步训练、6组pilot与3组完整100测试+3验证评分。最终选中 **A7750／原目标继续500步**，NR **2.31973328**、FR **1.44172329**、S_proxy **3.76145657**；相对父7250，NR+0.00660815、FR-0.00484572、S+0.00176243。本次同A100单卡新鲜配对的原工具预估 **58.45892→58.69787分**；固定本次父速度仅质量变化为**58.53038分**，均为经验预测。512×512 event p10 **46.34194ms**。本轮增益较小，约73%的NR收益被FR回退抵消；肉眼主要是细微纹理差异。全部103原PNG保留/cache；共享归档仅保存精确resume、可逐字节恢复raw的recipe、4例原图和小型证据，完成状态以新prefix `/temp/independent_sr_cuda_20261013_artifacts/nr_push4_20261014/DONE` 及[独立归档回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push4_backup_receipt_20261014.json>)为准。[完整中文报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_PUSH4_REPORT_20261014.md>) · [最新4例对比](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push4_visuals_20261014/index.html>) · [GT负例](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push4_val_visuals_20261014/index.html>) · [权重与续训位置](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push4_selection.json>)。30416端口，实验根目录 `/cache/independent_sr_cuda_20261013`。以下保留历史记录。

2026-10-14第三轮CUDA NR优化：三条分支从同一6500父模型分别新增750步，完成9组小样本及3组完整100测试+3验证评测。最佳为 **C7250／重锚6500／加强NR**，NR **2.31312513**、FR **1.44656901**、S_proxy **3.75969413**；相对上一轮6500，NR+0.03481477、FR−0.01487168、S净+0.01994309。新鲜同A100单卡配对测速p10 **46.82435ms**，原工具预估 **55.55522→56.41263分**；固定父模型速度仅质量变化为56.32852分，均为经验预测。MANIQA/CLIPIQA上升，MUSIQ/NIQE与FR下降；肉眼变化较小，不能解释为全部细节更真实。选中权重、匹配完整Adam/RNG续训状态和103原PNG共400文件（6,294,606,619 bytes）已由复制助手逐文件SHA回读后提交至共享存储 `/temp/independent_sr_cuda_20261013_artifacts/nr_push3_20261014/DONE`；独立最终核验以[归档回执](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push3_backup_receipt_20261014.json>)为准。[完整中文报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_PUSH3_REPORT_20261014.md>) · [4例原图和局部比较](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push3_visuals_20261014/index.html>) · [验证GT负例](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push3_val_visuals_20261014/index.html>) · [选型与权重路径](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push3_selection.json>)。服务器仍为30416端口，实验根目录`/cache/independent_sr_cuda_20261013`。以下保留历史记录。

2026-10-14第二轮CUDA NR优化：四条分支各新增250步，完整评测三组新模型（每组100测试+3验证）。最佳为**父6250保真锚点/MUSIQ训练权重1.5/global6500**，完整NR **2.27831036**、FR **1.46144068**、S **3.73975104**，相对父6250的NR增加0.01192532、S增加0.00964951。单张A100 512×512 p10 **46.35085ms**；原工具本次同卡校准预测 **55.76799分**，固定父延迟仅质量变化的预测55.69676（父55.32547），不是官方榜分。四终点/优化器/RNG恢复已备份，最终评分归档以新共享目录`/temp/independent_sr_cuda_20261013_artifacts/nr_push2_20261014/DONE`为准。[完整中文报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_PUSH2_REPORT_20261014.md>) · [新三版本可视化](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_push2_visuals_20261014/index.html>) · [选型和权重位置](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/results/nr_push2_selection.json>)。以下保留历史记录。

2026-10-14日间补测：原score_dir.py对6000/6250各103张独立复测，两组各418项指标与旧full逐值完全相同（最大差0）。最新[5例可视化比较页](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_recheck_visual_20261014/index.html>)已下载10张原4K PNG，含case83负例；[复测说明](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/nr_metric_recheck_20261014/RECHECK_REPORT_20261014.md>)。

2026-10-14最新：30416 CUDA端口已新增完成保守NR微调250步（global6250），完整100测试+3验证评分NR **2.26638504**、FR **1.46371649**、S **3.73010153**。同协议compiled6000基线NR2.25317108/S3.71710004。单卡A100 512×512 p10 **45.78048ms**；原工具同卡G4校准后预测 **55.5152分**，属于工具估计。权重、两个分支完整优化器/RNG状态及全部新评分PNG已经逐文件校验备份至 `/temp/independent_sr_cuda_20261013_artifacts/nr_microtune_20261014`。详见[本轮中文微调报告](</Users/hezihao/Documents/New project/csig_scoring_tool_review_20261013/score_search/NR_FINETUNE_REPORT_20261014.md>)。下方保留NPU及先前CUDA阶段记录；NPU截断PNG与CUDA round-PNG分数分别比较。

`render_search.py` 与 `score_search.py` 都支持 `--test-cases pilot|all`。
pilot 固定测试图为 `1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100`，始终包含验证图 1–3。pilot 仅输出 `S_subset`；all 包含测试 100 图并输出 `S_proxy`。两个字段不混用。

渲染保持模型初始化前 seed231、BF16、x1、patch512/stride256、空提示、上游 PIL uint8 截断量化、原尺寸 PNG。评分在每个数据集开始时仅 seed1234 一次，逐图执行 LPIPS/DISTS（仅验证）及 NIQE/MUSIQ/MANIQA/CLIPIQA；NR 用 FP32，FR 用 FP16 autocast。MANIQA 100 次裁剪、加权微批 4；各指标先取均值再合成分数。

以下为目录布局示例。候选目录里的 `data` 应指向 `${sr_search_root}/scoring_assets/data`；模型底座与 HYPIR 工程分别在 `models/Manojb__stable-diffusion-2-1-base`、`repos/HYPIR`，测试输入在 `assets/input`。指标 helper 为 `code/run_metric_study.py` 和 `code/metric_memory.py`；`csig_metrics.py` 放在 `code/score_search`。

```bash
sr_search_root=/cache/independent_sr_score_search_20261013
sr_candidate_dir=$sr_search_root/evaluations/s6000_mt200_ct200_pilot
sr_worker_python=$sr_search_root/env/bin/python

"$sr_worker_python" -u "$sr_search_root/code/score_search/render_search.py" \
  --project-root "$sr_search_root" --eval-root "$sr_candidate_dir" \
  --checkpoint "$sr_search_root/adaptation/runs/ascend_balanced/step6000.pth" \
  --test-cases pilot --model-t 200 --coeff-t 200 --backend npu --device-index 0

"$sr_worker_python" -u "$sr_search_root/code/score_search/score_search.py" \
  --tool-dir "$sr_search_root/code/score_search" --metric-code-dir "$sr_search_root/code" \
  --weights-root "$sr_search_root/scoring_assets/weights" \
  --test-dir "$sr_candidate_dir/render/test" --val-dir "$sr_candidate_dir/render/val" \
  --gt-dir "$sr_search_root/scoring_assets/data/验证集" \
  --label s6000_mt200_ct200_pilot --out-dir "$sr_candidate_dir/scores" \
  --test-cases pilot --backend npu --npu-index 0
```

CUDA 改为 `--backend cuda`。默认 `auto` 会在 CUDA 可用时选择 CUDA，否则选择 NPU；CUDA 分支不导入 `torch_npu`。单卡可见环境统一用逻辑设备 0。`--npu-index` 与 `--device-index` 是同一个参数的兼容别名。

每个 checkpoint、coeff_t、case 集合使用独立候选目录。渲染和评分分别保存完整 RNG 与原子进度检查点；恢复时检查源代码、输入/GT、模型或指标权重、PNG 哈希。不要把 pilot 目录切换为 all，也不要覆盖旧 8000 评测。由于推理 RNG 连续推进，pilot 某张图与 all 中同编号图的潜变量随机序列可能不同；仅在相同 case 集合和协议内比较候选。

评分结果在 `scores/result.json`，逐图指标在 `scores/measurements.jsonl`。这些脚本不测量时延、不生成预测榜分，不使用验证 GT 训练。HYPIR 的 VAE 会采样，时延测试应另进程进行，以免 warmup 改变渲染 RNG。

## 2026-10-13 完整质量结果

在30207的Ascend910B3上完成4组完整评分；每组100测试图、3验证图，418项逐图指标。

| 权重/coeff_t | 测试 NR | 验证 FR | S_proxy |
|---|---:|---:|---:|
| s6000_mt200_ct175_all | 2.25263586 | 1.46441658 | 3.71705244 |
| s6000_mt200_ct200_all | 2.24325909 | 1.44533869 | 3.68859777 |
| s8000_mt200_ct200_all | 2.21031602 | 1.45006194 | 3.66037796 |
| s10000_mt200_ct200_all | 2.21015056 | 1.45020688 | 3.66035744 |

当前最佳为step6000、model_t200、coeff_t175，比step8000/coeff_t200提高0.05667448。完整证据在 `results/s6000_mt200_ct175_all/`；9组pilot用于筛选，不能当作100图完整分。已完成PNG与评分证据保存在30207共享目录 `/temp/independent_sr_score_search_20261013_artifacts`，13组全部有artifact_complete标记，最后DONE已写入。原训练没有被更改或重启。

## 30416 CUDA 校准与复评

临时CUDA主机4张A100-SXM4-80GB；所有速度数据都用一张GPU0，同机G4参考、warm30+n200、CUDA event p10、FP16输入512×512，单独进程测速度。参考G4为0.65024000毫秒；HYPIR step6000/coeff_t175原GPU Runner为113.697995毫秒。固定空提示词缓存为105.327512毫秒，连续3次原生输入检查GPU输出逐bit、floor/round量化与所有RNG状态一致。cache+LoRA fuse为85.141502毫秒，融合版本必须单独评估质量。原始结果在 `cuda30416_calibration/` 和 `cuda_runner/cache_variants/`。

CUDA提交口径的质量生成正在独立重跑：工具输入`load_img(...).half()*2-1`，原尺寸4K，输出clamp后round到uint8、无损PNG；这与NPU原FP32输入/floor量化不同，不能直接混用质量分。两组CUDA目录为服务器 `/cache/independent_sr_cuda_20261013/evaluations/s8000_mt200_ct200_cuda_all` 与 `s6000_mt200_ct175_cuda_all`。完整质量与校准速度都齐备后再形成代理榜分预测。Torch2.8.0+cu128、PyIQA0.1.10，原评分脚本和权重保留原SHA。NumPy2兼容shim只恢复旧imgaug使用的dtype目录，不改变评分公式或RNG。

实时CUDA状态为服务器根目录 `results/status_cuda_search.json`，本地副本 `results/status_cuda_search.json`。

## CUDA 完整复评与最终选择（2026-10-13）

三组实际CUDA输出均已生成、完整评分和保存：每组100测试图+3验证图，418项逐图标量。全部309张原尺寸PNG已逐张SHA验证并保存到30416共享目录 `/temp/independent_sr_cuda_20261013_artifacts`。NPU质量与CUDA质量分开统计；下表全部使用CUDA工具提交口径的FP16输入/round PNG。

| 配置 | 完整代理S | 512×512 event p10 (ms) | 同机G4标定后的工具预测分 |
|---|---:|---:|---:|
| s8000_mt200_ct200_cuda_all / original | 3.65848450 | 113.29321 | 44.10945 |
| s6000_mt200_ct175_cuda_all / original | 3.71777519 | 113.69799 | 45.92732 |
| s6000_mt200_ct175_cuda_all / cached_empty_caption | 3.71777519 | 105.32751 | 46.63292 |
| s6000_mt200_ct175_compiled_cuda_all / cached_fused_compiled_tiles | 3.71710004 | 46.43932 | 54.86132 |

最佳预测配置是 **step6000 / model_t200 / coeff_t175 / 空提示词缓存 + LoRA融合 + UNet/VAE固定tile编译**。完整代理S为3.71710004、测试NR2.25317108、验证FR1.46392896；相对未编译6k175的3.71777519仅少0.00067515。其512时延46.43932ms，相对原6k175的113.69799ms减少59.16%；工具预测54.8613，比原8k200的44.1095多10.7519分（约24.4%）。这些是用户工具经验公式的估算，不是官方榜单实测；原五个轻量像素模型拟合的±2残差不能当成HYPIR的保证区间。

- 选择的可调用Runner目录：`/cache/independent_sr_cuda_20261013/submissions/hypir_s6000_mt200_ct175_cached_fused_compiled/model_dir`，包含`runner.py`和`config.json`。原配置/原结果未覆盖。
- 权重：`/cache/independent_sr_cuda_20261013/source_payload/adaptation/runs/ascend_balanced/step6000.pth`，SHA256 `2c82a71a94aa754bd4a26a2a474c2bbe4989bde1d38d2e15d9e58eb0c3e408bc`。同份权重还在30207持久化payload中。底模引用`/temp/zihaohe_stage/baselines/sd21b`，与30207 SD2底模逐SHA一致，三个评测目录均有base_model_provenance.json。
- 最佳103图与逐图指标：服务器`evaluations/s6000_mt200_ct175_compiled_cuda_all/{render,scores}`；本地`results/s6000_mt200_ct175_compiled_cuda_all/scores/measurements.jsonl`。
- 质量/速度联合结果：本地`results/quality_latency_comparison.json`，复现脚本`join_quality_latency.py`只接受完整103个病例、418标量、正确顺序和PNG哈希匹配，直接调用未修改的原`predict.py`。时延warm30+n200，与G4同一张A100 GPU0，质量在另进程生成，无latency warmup污染RNG。
- 原图样例：本地`cuda_outputs/`保存三版本各case7/16、共6张原PNG，`hash_proof.json`与生产者事务哈希逐项一致。额外四张NPU最佳完整输出在`best_full_examples/`，也有完整SHA证明。

实际视觉复核限于CUDA case7和16：主要文字、窗框与窗格骨架保留；6k175相对8k200的规则点纹/墙面蜂窝对比稍低，编译版仍保留这类伪纹，不能据这两例推断全103图或细节真实性。最终全量质量值与数值不同的编译版严格独立保存。历史原NPU结果S=3.71705244继续作为NPU协议结果保留。

原工具122项源码/权重/数据、全309PNG与GT的完成后重hash及逐项聚合审计见`audits/`。MANIQA离线初始化只取消随后被完整原checkpoint覆盖的额外预训练下载，最终316个状态和152个ViT条目与原包逐键/shape/dtype/value一致；前向和100-crop计算未改。当前stream恢复对历史资产的fingerprint覆盖较弱，最终审计通过后才发布本次完整结果；未来中断续评须重新核对外部底模、所有指标权重、GT与已提交行顺序，不直接套用变化后的资产。
