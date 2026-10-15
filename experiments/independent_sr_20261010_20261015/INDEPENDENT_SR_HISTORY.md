> 本次汇总的当前现场状态以[总交接](README.md)和[资源快照](archive/server_cuda30416/LIVE_RESOURCE_SNAPSHOT.json)为准；本页中历史阶段的在线性、缺失资产与旧选择仍按来源时间解释。

# 独立 SR 原始模型、伪标签微调与昇腾迁移历史（2026-10-10—13）

整理日期：2026-10-15。本页来自本地已保存报告与 JSON，供 `RedRoman/csig-track2-tinyenhance` 的 `experiments/independent_sr_20261010_20261015/` 整理使用。本次仅做本地只读取证与新文件整理，没有连接实验服务器、训练、重新评分或执行 GitHub 操作。这里的“完成”“生成中”“等待”均指来源快照时刻，不能当作端口当前可用性、当前进程或当前最佳模型。

本页覆盖早期独立 SR 对比、23 版评分、HYPIR/PromptIR/VOSR2 伪标签适配、历史多教师图重评、昇腾训练与 CUDA 迁移入口。2026-10-14—15 的 CUDA Push1—7、最新 TensorRT 后端、缓存重置恢复及 PixG5i 蒸馏应由各自新报告衔接，不能用本页替代其最终选择记录。

## 1. 可核查的本地来源

| 来源 | 证据范围 | 此次处理 |
| --- | --- | --- |
| [第二轮完整中文报告](history_sources/round2/实验报告_指标与微调及视觉案例.md) | 2026-10-10 16:21:51 北京时间快照；23 版指标、训练配方、视觉与多教师历史 | 原文件 **41,546 bytes**，逐字节复制；未改写正文 |
| [23 版机器可读表](independent_sr_round2_versions.json) | 从报告第 2 节转录，保留原表四舍五入精度 | 新结构化索引；不是重新测量 |
| [昇腾历史索引](history_sources/ascend/ASCEND_STATUS_20261012.md) | 昇腾原始阶段及随后追加的 CUDA 历史摘要 | 原字节复制；文件顶部追加过后续状态，**不能作为最新模型指针** |
| [昇腾固定 12 图分项](history_sources/ascend/native_subset_0_1000_4000_8000.json) | 0/1000/4000/8000 步同口径子集对照 | 实际 JSON 在本地存在 |
| [昇腾 8000 步全量摘要](history_sources/ascend/full_score_s8000/score_summary.json) | 100 test + 3 val；原工具顺序无损 PNG 评分 | 连同状态、协议、manifest、result、复核记录复制 |
| [8000 步独立本地复核](history_sources/ascend/full_score_s8000/local_verification.json) | 重算 NR/FR/S、103 记录与 8 个指标权重哈希 | 实际 JSON 在本地存在 |
| 原仓库 `README.md`、`docs/CSIG_SUBMISSION_LEDGER.md`、`docs/memory/project_csig_track2.md` | 2026-08—09 比赛模型、提交与协议变更历史 | 保留既有仓库文档，不复写其旧结论 |
| 原仓库 `docs/AUTODL_MIGRATION_README.md` | 2026-08-19 VOSR/TinySR ModelArts→AutoDL 迁移 | 仅引用相关路径、版本与已验证数值 |

全部已复制小文件和既有仓库来源的实际 SHA256/大小见 [来源清单](independent_sr_history_source_manifest.json)。原目录定位分别是 `independent_sr_round2_review_20261011/`、`independent_sr_ascend_review_20261013/`、`csig-track2-tinyenhance/`，均位于本地工作区 `/Users/hezihao/Documents/New project/`。

历史用户交接入口为 `/cache/independent_sr_round2_20261010/reports/progress_20261010/AGENT_HANDOFF.md`。**本地这次未找到该文件本体**，不声称已重新读取其正文；本页以实际存在的完整中文报告为证据。报告引用的 `model_metrics.csv`、`case_metrics.csv`、`validation_metrics.csv`、`snapshot.json`、`case_evidence.json`、`measurements/` 与 `legacy_teachers/manifest.json` 也未在这份最小本地报告目录中找到，不能将这些引用当作已上传附件。

## 2. 原始独立实验的位置与端口沿革

| 阶段 | 历史端口信息 | 历史实验根目录/说明 |
| --- | --- | --- |
| 早期 ModelArts VOSR/TinySR | 30138，后 30674 | `/cache/zihaohe/csig_isr`；见既有项目记忆 |
| 2026-08-19 AutoDL 迁移 | 24686 | 数据盘 `/root/autodl-tmp/zihaohe`，通过 `/cache/zihaohe` 软链兼容旧代码 |
| 2026-10-10 独立 SR round2 | 用户交接先给 32316，随后更换 32193 | `/cache/independent_sr_round2_20261010`；端口为对话历史，不代表现在仍挂载此目录 |
| 昇腾 HYPIR 新阶段 | 文档明确 32193；后续参数搜索记录为 30207 | `/cache/independent_sr_ascend_20261012`；实际评分后端按协议为 Ascend NPU |
| CUDA 迁移 | 文档明确 30416 | `/cache/independent_sr_cuda_20261013`；历史记录为 4 张 A100-SXM4-80GB |

用户还曾授权尝试 31320、31051、31731、30794、30213 等端口。这里只记录授权沿革，不将“可连接”写成“有 CUDA”“有空卡”或“旧权重仍存在”。昇腾报告明确原 A100 31904 已关闭，当时无法读取旧断点，因此重新用公开 HYPIR 初始化；该阶段不是对旧 1000 步的精确恢复。

round2 的历史路径布局：

```text
/cache/independent_sr_round2_20261010/
  code/                                  训练、推理、评分执行器
  assets/input/case1.jpg ... case100.jpg   test100 原始输入
  assets/validation/                     3 对验证 LQ/GT
  raw_outputs/<label>/                   模型原始输出
  assets/outputs/<label>/                 测试评分用 JPEG
  assets/validation_outputs/<label>/      验证评分用 JPEG
  adaptation/teachers/<policy>/selection.json
  adaptation/runs/<run>/config.json       训练配置
  adaptation/runs/<run>/                  权重、完成记录
  review_previews/                       用户交接指定的放大对比位置
  results/<label>/measurements.jsonl      逐图逐指标、种子与耗时
  results/<label>/summary.json            汇总
  results/status_*.json                   阶段状态入口
  logs/                                  实际运行日志
```

这些是历史定位，**本次未远端检验目录存续**。用户另指出 `adaptation/runs/nr_v2_strength/` 与 `adaptation/runs/nr_v2_balanced/` 两组训练；本次对上述本地报告、相关配置/状态摘要与仓库文档检索，未找到足以确认这两组具体目标、步数、检查点、完成状态或分数的证据。因此仅保留路径线索，状态为“历史待核实”，不能给它们套用后面的 `ascend_balanced` 或 CUDA 分数。

## 3. round2 的 23 个完整评分版本

数据快照为 **2026-10-10 16:21:51（北京）**。下表直接摘取原报告，均为报告显示精度；不是本次重新评分。测试 100 图没有 GT，NR 由四项无参考指标构成；FR 仅来自另外 3 对有 GT 验证图。表内 S 不含速度项。官方分仅两条原报告明确记录，其余为空，不能用本地代理分冒充官方榜分。

| 版本 | NR↑ | FR↑ | S_proxy↑ | 报告已知官方分 |
| --- | ---: | ---: | ---: | ---: |
| `final4h_balance` | 2.6973 | 1.3216 | 4.0189 | 92.1755 |
| `pushv2_teacher_quality` | 2.4588 | 1.3837 | 3.8424 | 75.6735 |
| `hypir_sd2_t200` | 2.1155 | 1.4434 | 3.5589 | — |
| `seesr_50step` | 2.0499 | 1.4635 | 3.5133 | — |
| `hypir_perceptual_select_s1000` | 1.8511 | 1.4579 | 3.3090 | — |
| `hypir_perceptual_select_s250` | 1.8171 | 1.4459 | 3.2630 | — |
| `invsr_3step` | 1.7895 | 1.3103 | 3.0998 | — |
| `hypir_nr_select_s250` | 1.6622 | 1.3905 | 3.0527 | — |
| `hypir_nr_select_s1000` | 1.6690 | 1.3830 | 3.0520 | — |
| `realesrgan_x4_native` | 1.6378 | 1.3289 | 2.9667 | — |
| `realesrgan_x2_native` | 1.6235 | 1.3403 | 2.9639 | — |
| `vosr2_nr_select_s250` | 1.4607 | 1.3873 | 2.8481 | — |
| `real_hat_gan_native` | 1.4763 | 1.3141 | 2.7905 | — |
| `vosr2_public_native` | 1.3544 | 1.3847 | 2.7391 | — |
| `vosr2_nr_select_s1000` | 1.3596 | 1.3401 | 2.6998 | — |
| `supir_q_50step` | 1.4630 | 1.1417 | 2.6047 | — |
| `foundirv2_20step_native` | 1.2254 | 1.2826 | 2.5080 | — |
| `promptir_nr_select_s1500` | 1.1777 | 1.2988 | 2.4765 | — |
| `promptir_public_native` | 1.1355 | 1.2787 | 2.4142 | — |
| `promptir_perceptual_select_s1500` | 1.1144 | 1.2811 | 2.3954 | — |
| `promptir_nr_select_s500` | 1.1073 | 1.2843 | 2.3916 | — |
| `promptir_perceptual_select_s500` | 1.0971 | 1.2799 | 2.3770 | — |
| `input_unmodified` | 1.0968 | 1.2802 | 2.3770 | — |

主要结论：旧 `final4h_balance` 的质量代理 S 为 4.0189，对应报告已知官方 **92.1755**，并非缺失的 92.9855 提交件。新公开原版中 HYPIR 的 S 最高，SeeSR 的 FR 最高。HYPIR 两种伪标签微调均未提高完整 S；PromptIR 的 NR 教师 1500 步相对原版 +0.0624；VOSR2 NR 教师 250 步 +0.1090，而 1000 步反而 -0.0393。不能把训练更久或局部更锐写成整体进步。

### 3.1 原报告评分协议

```text
NR = mean(CLIPIQA_test100) + mean(MANIQA_test100)
     + mean(MUSIQ_test100)/100 + max(0,(10-mean(NIQE_test100))/10)
FR = 2 - mean(LPIPS_val3) - mean(DISTS_val3)
S_proxy = NR + FR
```

PyIQA 0.1.10；MANIQA 使用 KonIQ 权重，不是 PIPAL。每张图、每个指标分别重置 seed1234；NR FP32，LPIPS/DISTS FP16 autocast；TF32 与 cuDNN benchmark 关闭。MANIQA 每图 100 crops、microbatch4、按权重聚合；MUSIQ 保留原生 token，分块减显存。旧 JPEG 保留原字节，新 PNG 则转为 JPEG quality100/subsampling0 后评分；若输出尺寸不同，先 Lanczos 回到输入尺寸。

这些设置与后来 **无损 PNG、按 split 连续 RNG、CUDA 原工具输入量化顺序及后端** 不完全相同；跨协议的数值不能直接当成训练增益。主表未测 PSNR/SSIM，不能补造；legacy 小节另测验证图的 PSNR/SSIM不改变主表范围。

### 3.2 原版测试关键设置

| 模型 | 本轮实际设置 |
| --- | --- |
| HYPIR | SD2.1-base + rank256 LoRA，model_t=coeff_t=200，patch512/stride256，空 caption，seed231 |
| SeeSR | 50 steps，CFG5.5，process512，RAM/DAPE，upscale1 |
| FoundIR-v2 | 20 steps，CFG5，RealVisXL4，固定通用 caption，wavelet；未部署逐图 LLaVA |
| VOSR2 | one-step，x1，DiT tile768/overlap128、VAE tile512/overlap64 |
| PromptIR | DeepInv 公开权重，官方 decoder=True 架构，x1 tile256/overlap64，BF16 |
| InvSR | SD-Turbo，3/5 steps，chop256/batch1，wavelet，原生 x4 后缩回 |
| DiffBIR | v2.1，30 steps，x1，无 captioner，tiled |
| SUPIR | Q/F，50 steps，x1，无 LLaVA，tiled VAE |
| RealESRGAN / RealHATGAN | 公开 x2/x4 权重，输出回到任务输入尺寸 |

具体公开工程链接、提示词、全六指标和逐图表保留在完整中文报告，不将这些有限配置的结果称为相应模型的性能上限。

## 4. 伪标签微调的真实目标与检查点

两种教师策略都只使用 test100 输入与预先固定的教师伪标签，验证 3 图不训练；不是测试 GT。`nr_select` 实际选中旧 final4h 95 张、HYPIR 2 张、pushv2 3 张；`perceptual_select` 实际 HYPIR 62 张、SeeSR 38 张。后者只是实验标签，并非真实感知 GT。教师库没有随着学生输出递归改写。逐图选择先 clamp 每图 NIQE 项，与最终“均值后 clamp”的聚合并不完全一样。

共同设置：seed20261010，AdamW/weight_decay0，BF16 autocast，grad clip1；输入和教师同位置 crop、同步水平/垂直翻转。

| 训练对象 | 可训练部分/训练设置 | 损失 | 保存点与服务器历史位置 |
| --- | --- | --- | --- |
| HYPIR（两策略独立初始化） | rank256 LoRA、514 张量；batch2/crop256/lr2e-6；1000 步；VAE encode mode/no_grad、UNet checkpointing | L1 + .05 LPIPS + .1 gradient | `adaptation/runs/hypir_adapt_<policy>/step{250,1000}.pth` |
| PromptIR（两策略独立） | 全参数，batch4/crop192/lr1e-5；1500 步 | L1 + .1 gradient | `adaptation/runs/promptir_<policy>/step{500,1500}.pth` |
| VOSR2（nr_select） | 末 6 DiT blocks + final_layer；其他 DiT/VAE/DINO 冻结；batch1/crop256/lr1e-6；1000 步 | RGB L1 + .1 latent L1 + .05 RGB gradient | `adaptation/runs/vosr2_adapt_nr_select/step{250,1000}/ema_model.safetensors` |

HYPIR 训练入口为历史 `train_hypir_adapt.py`，PromptIR 为 `train_aio.py`/`infer_aio.py`，VOSR2 为 `train_vosr2_adapt.py`。原报告链接到上层代码快照；此次最小本地报告目录不含这些源码，不能宣称本次已重新静态审计或跑通。

PromptIR 权重 HF revision 为 `9c27b4db627f3fb725166ae6366696e4e61349e9`；VOSR2 为 `f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3`，使用原配 Qwen-Image 2D VAE 与 DINOv2-L。不要将另一条早期使用 SD2.1 VAE 的 VOSR 自研训练线混成同一模型。

报告快照仍未完成的三项：DiffBIR 68/103、InvSR 5 步 28/103、SUPIR-F 43/103。仅是当时已生成 PNG 数，**不是现在仍运行，也不证明后续已完整评分**。MoCE-IR 当时未取得原权重，只有准备配方，没有可写成成果的训练或评分。

## 5. 历史多教师图重评与视觉证据

2026-09-19 保存的 `legacy_scene_pretext/final` 是百图教师 PNG 原件，非 round2 HYPIR/SeeSR 选图库、也非新学生推理图。默认 BSRGAN_s4；部分 case 使用 BSRNet/BSRGAN 配比；case7/23 额外融入 MARCONet++ 文字候选。不可靠文字结果被拒绝，未声称对全部 100 图穷举最优组合。

重算完整 100 图 NR：final **1.733204**，pretext **1.733370**，加入文字专项后的 NR 变化为 **-0.000166**。这不高于当轮公开 HYPIR/SeeSR。A/B/C 各 10 张、D 5 张、E 1 张等候选，不能拿不同样本数均值直接排全量名次；历史 3 张验证图的 8 种策略与百图教师没有唯一配对，不能任意拼 NR 与 FR 得出虚构完整 S。

历史服务器定位：新评分快照教师图在 `/cache/independent_sr_round2_20261010/legacy_teachers/images/`；原教师原件记录为 `/root/autodl-tmp/independent_sr/30_our_work/runs/finalize_scene_teachers_20260919/images/`。本次没有远端确认其是否仍存在。

本次实际检查并原字节保留以下 3 张代表性拼图，总 **2,608,429 bytes**，未锐化、重排或重新导出。每张含同一区域的输入、旧版、HYPIR/SeeSR、微调与其他公开模型；图内分数是整图单图 NR，不是 ROI 分，也不是官方分：

| 图 | 内容/复核重点 | 大小 |
| --- | --- | ---: |
| [case07](history_sources/round2/visuals/case07_comparison.jpg) | 施工横幅、脚手架、文字与直线 | 923,692 B |
| [case23](history_sources/round2/visuals/case23_comparison.jpg) | 快餐招牌文字、笔画与树枝交叉 | 749,581 B |
| [case41](history_sources/round2/visuals/case41_comparison.jpg) | 花瓣/叶片，检查生成纹理与形态 | 935,156 B |

完整报告引用的 case18/25/28 及 legacy 拼图已由总归档补齐，13张原字节图片均在 `history_sources/round2/visuals/`。历史 CSV/部分源码仍缺失，见总览，不使用无关新图或新结果补造。未发现可确认为 `nr_v2_strength/balanced` 的对应图片，不能给上述拼图改名冒充这两组。

## 6. 昇腾部署、NR 阶段训练与风险

原 A100 关闭后，在 `/cache/independent_sr_ascend_20261012` 从公开 HYPIR 重新初始化。原 CANN 缺少完整 910B 算子，补齐项目内 `runtime/opp`、匹配 OPP 配置/实现、`runtime/hostlibs/data`、`runtime/share/info/runtime/version.info`；未修改系统目录。调度器覆盖 CANN Python 路径造成 `tbe` 导入失败后，保留正确路径才恢复。

报告记录实际 BF16 512 矩阵、卷积、注意力前后向有限且非零；首步 NR 输入梯度 MUSIQ .331058、MANIQA .209434、CLIPIQA 2.338521。CLIP 梯度量级明显更大，训练加权项不能等同正式 NR 等权聚合。首次编译/加载约125秒、随后单步约1.16秒仅是早期测量，不是稳定吞吐保证。

训练目录为 `adaptation/runs/ascend_balanced/`，检查点 `stepN.pth`，阶段状态 `results/status_ascend_train.json`，逐步状态 `results/status_ascend_balanced.json`/`train.jsonl`，预览 `review_previews/`，逐图指标 `review_previews/ascend_balanced_sSTEP/metrics_native_ascend_v1.jsonl`。记录按1000步段实际查看预览再决定是否继续；受控续训批准不等于无伪影发布批准。

| 历史阶段 | 已记录变化/结论 |
| --- | --- |
| 1000→5000 | lr5e-7、reward .05 的受控续训；文字形变、花瓣蜡感、植物过度生成持续存在，NR 非单调 |
| 5000→6000 | 因子集停滞改 reward .10，lr5e-7；6000 NR 升高但玻璃规则点阵/蜂窝与墙面/天空颗粒更明显 |
| 6000→7000→8000 | 改 lr2e-7、reward .075、gradient anchor .3（原 .1），保持原冻结锚点、像素/LPIPS项；7000规则点阵减轻但仍残留 |
| 后续历史摘要 | 记录已保存到10000步，再用6000/8000/10000断点做参数筛选；不把旧等待器/PID当作当前运行任务 |

实际保存的固定 12 图是 case7/16/17/18/23/24/25/28/32/33/41/44。JPEG100/subsampling0、每图 seed1234、Ascend FP32、MANIQA100 的 **NR_subset**：0步2.13353549、1000步2.26670725、4000步2.21651333、8000步2.26164816。这是逐图 clamp 后求均值，与完整工具“先均值再 clamp”、无损 PNG 的结果不同；子集分不得标为全100图或官方成绩。跨 CUDA 数值等价未建立。

历史关键检查点备份记录：1000步 SHA256 `34f6f3619f44184c08be10c80b973fc61512cd0fe45fd7e0336956fc6d3f3fcf`，5000步 `edab44c9d3e4df69f16b79e323faa473f560c52f23d1bcb42353a667b06e72bd`，各1,038,080,515 B；文档指向当时 Windows `D:\competitions\KEY_CHECKPOINTS_ASCEND_20261012`。本次没有访问该磁盘，因此这些是来源记载，非新完成的备份核验。

### 6.1 8000步原工具顺序全量评分：本地证据齐全

| 项目 | 实际 JSON 值 |
| --- | --- |
| 模型 | `ascend_balanced_s8000` |
| test/val | 100 / 3，共103记录 |
| NR | **2.210316020381451** |
| FR | **1.4500619371732077** |
| S_proxy | **3.6603779575546587** |
| NIQE / MUSIQ / MANIQA / CLIPIQA | 3.8264845276 / 43.1515911293 / .4724397105 / .6890088513 |
| val LPIPS / DISTS | .3483479818 / .2015900811 |
| 工具总分预测 | **null**：没有 AutoDL 校准的 CUDA512 event-p10，不能补成一个榜分 |
| 后端 | Ascend NPU；CUDA 数值等价未建立 |

checkpoint 为 `/cache/independent_sr_ascend_20261012/adaptation/runs/ascend_balanced/step8000.pth`，SHA256 `1446f3bc67f2d5743a9f00eb52c92d1c610124779a37a98646d2c1f7f3775225`。结果历史定位为 `evaluations/csig_step8000_20261013/results_png_tool_order/result.json`。本地独立复核重算 NR/FR/S 相同、103记录、8个指标权重哈希匹配。

此全量协议为 `csig_png_npu_stream_tool_order_v1`：原生无损 PNG、RGB FP32[0,1]、不 resize，seed1234 **每 dataset/model 一次，先 val 后 test 连续消费**；不是前述固定12图每图重置协议。PyTorch2.7.1+cpu/torch_npu2.7.1.post2/PyIQA0.1.10。protocol/result 保留实际输入、GT、输出及指标权重哈希；本页不重新装权重或再次计算模型指标。

## 7. 参数搜索与 CUDA 迁移接口

昇腾历史摘要明确 30207 上完成4组完整100+3：6000/coeff175 S3.71705244、6000/coeff200 S3.68859777、8000/coeff200 S3.66037796、10000/coeff200 S3.66035744。model_t200/coeff_t175 的6000步较8000/coeff200高 .05667448；另外9组pilot只用于筛选。来源记载全部13组已校验归档于 `/temp/independent_sr_score_search_20261013_artifacts`，本次不重新校验归档字节。

迁至30416 CUDA A100后，原始512单卡推理约113ms；记录的缓存→LoRA融合→tile编译减少到46.43932ms。缓存版曾有输出/RNG完全一致证明；融合/编译属于另一个需重新验证数值及质量的后端，不能把早期 NPU FP32/floor 质量直接沿用为 CUDA FP16/roundPNG 质量。

随后 CUDA 原生103图完整重评的6000/coeff175 S为 **3.71710004**，该历史批次 p10 **46.43932ms**、原工具经验预测 **54.8613**；8000同批同协议预测44.1095。它们不是官方分数，也不能与2026-09旧 AutoDL拟合分或后续不同 G4归一化批次直接拼接。更晚的6250/6500/7250/7750训练、同权重优化与TensorRT后端在其他报告中继续，本页不把旧索引最顶端“当前”字样解释为此刻状态。

## 8. 更早 VOSR/TinySR 迁移及旧仓库的相关证据

`docs/AUTODL_MIGRATION_README.md` 记录2026-08-19从30674的 `/cache/zihaohe` 迁到AutoDL数据盘 `/root/autodl-tmp/zihaohe`，保留软链 `/cache/zihaohe -> /root/autodl-tmp/zihaohe` 使旧绝对路径继续生效；`_relink.sh` 可恢复软链，`_env_autodl.sh` 设置项目/cache路径。所移 venv 为 `/cache/zihaohe/csig_isr/venv`，Python3.10、Torch2.5.1+cu124、torchvision0.20.1+cu124；其解释器软链有宿主环境依赖，不能把搬运目录直接当成通用可复现环境。

当时两端同口径校验：`train_out/vosr_iqa1.0/best.pth` 验证全尺度S两端均3.8957，test100 NR均2.4733；TinySR `train_out/combo_exact/best.pth` 验证S3.9286→3.9289、testNR2.5867→2.5866。它们是8月历史模型，不是10月 round2 VOSR2 或新HYPIR评分。

旧布局涉及 `VOSR/`、`VOSR_weights/`、`csig_isr/TinySR/`、`csig_isr/v3/`、`csig_isr/sd21base/`、`csig_isr/train_out/`。官方VOSR的 Qwen-Image VAE 链在源端已断，旧自研 `train_vosr_csig.py` 使用 SD2.1 VAE因而不受影响；不能据此认定官方VOSR推理依赖齐全。迁移材料另有MANIQA-PIPAL历史权重，**不得替换本轮KonIQ指标权重**。

既有提交台账最终记录2026-09-19 `pixG4_g` 实际官方 **152.9531**；G5c/G5i的约154为当时预测，并非已证明的官方成绩。它们与 round2 中 `final4h_balance` 的92.1755来自不同版本/阶段；与10月CUDA的工具预测也不是同一个表。台账中“当前最佳”“天花板”“不可达”等都是当时结论，后来已有继续实验，不应复制成全项目现状。

原仓库保留 `train_pix.py/pix_model.py/eval_pix.py`、`train_distill.py/eval_ck.py`、`csig_isr/eval_out/`、`pix_runs/`、`autodl_pix_out/`、`viz_*/` 与提交台账；原模型目录的历史定位为 `csig_isr/submit_pixL2d_g`、`submit_pixG4_g`、`submit_pixG5c_g`、`submit_pixG5i_g`。仓库 README 明确不包含数据集、教师图、特征缓存、大权重、提交zip、第三方repo/环境，不能仅凭有目录说明就声称GitHub能直接离线推理。

## 9. GitHub 整理范围与保留的不确定项

本 draft 连同 `history_sources/` 的13个实际小文件总2,848,039 B，可整体纳入新实验目录；代表图合计低于4MB。历史原报告保持原字节，完整23行来自原报告，并附可机器读取的全部六指标。源 SHA清单用于核对拷贝，不能代替服务器运行证据。

不确定项保持显式：`AGENT_HANDOFF.md` 本体、两组 `nr_v2_*` 的配置/终态/完整分数、DiffBIR/InvSR5/SUPIR-F在快照之后是否完成、原远端各路径当前存续、Windows旧备份是否仍在、当前端口/空卡状态。无证据的字段应留空或标历史待核实，不用邻近模型分数填补。

本地旧源与旧仓库未修改；只创建新staging资料。未读取凭据、未生成SSH配置、未进行远端或GitHub操作。原仓库 `docs/memory/` 相关历史引用保留，不进行整体覆写。
