> 本次汇总的当前现场状态以[总交接](README.md)和[资源快照](archive/server_cuda30416/LIVE_RESOURCE_SNAPSHOT.json)为准；本页中历史阶段的在线性、缺失资产与旧选择仍按来源时间解释。

# HYPIR 教师 NR 优化历史与证据草稿

> 文档用途：供本项目 GitHub 实验档案合并。仅整理 `csig_scoring_tool_review_20261013/score_search/` 下的必要中文报告、现有小型审计/状态/选择文件；本次未连接服务器、GitHub，未重新训练、评分、测速，未读取或上传大权重。它是有来源哈希的历史提取，不是一次新的资产完整性审核。生成时间：2026-10-15T13:44:27.099546+00:00。

## 最新可证实的 HYPIR 教师

截至本地选择记录 `2026-10-15T08:44:00.460157+00:00`，HYPIR 当前指针是 **末端 LoRA NR 微调 100 步的 s7850**，而非旧 A7750，也不是轻量 Pix 学生。它随后被固定用于 PixG5i 伪标签蒸馏；后者是独立任务，不改写 HYPIR 本身的训练史。

| 项目 | 当前记录 |
|---|---|
| label | `push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all` |
| raw SHA256 | `23de1a93b1ddd72a268788bebbdd8ba348b2819bc35cfa6a4501ffc822fdc06c` |
| immutable resume SHA256 | `35e339656581c89c772e6d9b00592a73cd397481f00d4d9bfd78c4bab7d05745` |
| test100 NR / val3 FR / S | **2.3267079146643903 / 1.4491248528162637 / 3.775832767480654** |
| 同轮 A7750 基线 S | 3.760926498369879；候选 ΔNR +0.0073691028、ΔFR +0.0075371663、ΔS +0.0149062691 |
| 512 单卡 event p10 | 候选 **32.7895050049 ms**；同轮基线 32.8038406372 ms |
| 原工具经验预测 | 同轮共同保守 G4 标定：**60.5694346386 → 61.2037920540**；尚未达到 65 |
| 视觉验收 | 固定 7/16/41/83 原尺寸 1:1 ROI，历史独立审阅和 root 均未见明显新增大几何变形或整片涂抹；不能推广为全 103 图无伪影 |

选择、全量质量审计、正式联合审计的本地小文件 SHA 已在本次整理中互相核对一致：

- `nr_push7_20261015/results/nr_push7_lastres_lr5e6_selection.json`
- `nr_push7_20261015/evaluation/lastres_lr5e6_s7850_trtunetvae_recovered_matched.json`：SHA `4c8afbffd2a47802e52b5d3f25a185bc50bb5d279d9dcc14ec0fc3f5c63467a9`
- `nr_push7_20261015/evaluation/recovered_lastres_formal_cpu.json`：SHA `484174d65832f41abe719d7f5b697cc07fc661e7319321275201074433b94909`
- `nr_push7_20261015/visuals/hypir_lastres_lr5e6_trt_full_final/preview_provenance.json`：SHA `fb090f77e7012034a359aed3cbdf0427fad46fcad390971be97dfd92c81f9105`

服务器路径是历史/恢复后的运行定位，不代表本次已重新检查其在线可用性：

```text
SSH port: 30416
experiment root: /cache/independent_sr_cuda_20261013
model_dir: /cache/independent_sr_cuda_20261013/submissions/hypir_push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae/model_dir
raw: /cache/independent_sr_cuda_20261013/adaptation/runs/hypir_push7_A7750_lastres_lr5e6_nr100/step7850.pth
resume: /cache/independent_sr_cuda_20261013/adaptation/runs/hypir_push7_A7750_lastres_lr5e6_nr100/resume_step7850.pt
full: /cache/independent_sr_cuda_20261013/nr_push7_recovered_full_evaluations/push7_a7750_lastres_lr5e6_nr_s7850_trtunetvae_all
```

## 分数与测速口径

**质量 S、工具预测分和官方榜分必须分列。** 原工具质量合成记录为：

```text
NR = mean_test(CLIPIQA) + mean_test(MANIQA) + mean_test(MUSIQ)/100
     + max(0, (10 - mean_test(NIQE))/10)
FR = 2 - mean_val(LPIPS) - mean_val(DISTS)
S_proxy = NR(test100) + FR(val3)
```

正式全量是 100 测试图 + 3 验证图、103 PNG / 418 标量。固定 pilot 是 20 测试图 + 3 验证图、23 PNG / 98 标量，只有 `S_subset`。由于 HYPIR posterior 采样连续推进，同 case 的 pilot PNG 不能替代 full PNG；相同模型在不同队列中也可能不是同一随机样本。

原工具经验预测使用原 `predict.py`（报告所载 SHA `0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1`）：

```text
g = candidate_event_p10_ms * 0.431 / same_batch_G4_event_p10_ms
prediction = 152.95 * 2**(S - 4.0841)
             * ((0.49 + 2.17*0.431)/(0.49 + 2.17*g))**0.2
```

时延是单卡 A100 的 **512×512 FP16 输入完整 infer**，独立 fresh 进程，seed231、warm30、n200、CUDA event p10；不是 4K 原图耗时。正式质量渲染不先测速热身，初始化前 seed231，连续 val3→test1..100；评分每 split 起始 seed1234 后连续执行，NR FP32、FR FP16 autocast，MANIQA 正式 100 crops。NPU 早期 FP32/floor PNG 与 CUDA FP16/round PNG 分开；不得直接拼成一个统一排名。

最新恢复后采用预先固定的 `min(G4_pre.p10,G4_post.p10)` 作为双方共同保守参考。本轮 G4pre=0.7474176288 ms、G4post=0.6492159963 ms。若单用 pre，两方估分是 62.2807192 / 62.9329909；正式采用共同 min 后是 60.5694346 / 61.2037921。不能从历史挑更有利 G4 或速度与当前质量拼接，也不能把历史 62.9537 与当前 61.2038 直接相减归因于质量变化。经验拟合不是官方榜单实测，对 HYPIR 没有被证明的 ±2 分保证。

## 从原始参数搜索到 A7750 的主线

下表保留各报告当时的完整质量口径与各轮测速，不构成跨批次预测分排名。早期部分值为报告打印精度；精确来源/证据等级见机器表。

| 阶段 | 版本与变化 | NR / FR / S | 当轮时延与预测 |
|---|---|---|---|
| 10-13 NPU 参数搜索 | step6000, model_t200, coeff175；优于 step8000/10000 coeff200 | 2.25263586 / 1.46441658 / 3.71705244 | 30207 Ascend 910B3；无同口径 CUDA 速度结论 |
| CUDA 原实现 | step6000/coeff175 | S 3.71777519 | 113.69799 ms，45.92732 |
| CUDA 缓存/融合/编译 | 空提示缓存、LoRA fuse、固定 UNet/VAE tile compile | 2.25317108 / 1.46392896 / 3.71710004 | 46.43932 ms，54.86132；像素改变后独立评分 |
| 第一轮 NR 微调 | conservative 从6000新增250到6250 | 2.26638504 / 1.46371649 / 3.73010153 | 45.78048 ms，55.51522 |
| Push2 | 6250锚点，MUSIQ权重1.5，新增250到6500 | 2.27831036 / 1.46144068 / 3.73975104 | 当轮46.35085 ms，55.76799 |
| Push3 | 从6500分叉，C加强NR，新增750到7250 | 2.31312513 / 1.44656901 / 3.75969413 | 当轮46.82435 ms，56.41263 |
| Push4 | C7250继承，A原目标再500到7750 | 2.31973328 / 1.44172329 / 3.76145657 | 当轮46.34194 ms，58.69787 |
| Push6 | 同A7750权重，无CL的RO+wavelet compile | 同A7750；103实际新PNG逐字节全等后桥接旧418指标 | 同轮46.9421→41.4382 ms，56.4358→57.8501 |
| Push7 同权重TRT | FP16 VAE，再加独立FP16 UNet引擎 | 最终同权重S 3.76092650 | 历史同批32.56105 ms，62.95370；不同批次不可直接混比 |
| Push7 受限NR | 仅最后残差块conv1/conv2和conv_out的6张LoRA更新100步，独立新UNet引擎 | **2.32670791 / 1.44912485 / 3.77583277** | **恢复后32.78951 ms，61.20379** |

训练支线数不能当作一条累计步数相加。Push2 四分支各250；Push3 三分支各750；Push4 三分支各500、D额外250。它们各有自己的优化器/RNG谱系。早期三个 NR 网络共享正式评分来源，训练 test100 输入参与无GT适应与选型，validation GT 未训练，因此这些结果不能宣称独立测试泛化。

## 训练设置与最新受限更新

6000→6250 开始 CUDA 训练使用 BF16 底模、514个FP32 rank256 LoRA张量；batch1随机512crop，初始新AdamW，conservative LR2e-7/reward.075/CLIP.1/gradient.3。保真项为同latent冻结锚点的 L1+.1LPIPS+.3梯度；训练MANIQA20crop，NIQE当时只评测。Push2以父6250重锚，MUSIQ1.5；Push3 C为teacher6500、reward.15/CLIP.5/MUSIQ1.5/LR2e-7；Push4 A沿此目标继承Adam到1750。MUSIQ训练权重提高并未保证正式MUSIQ上升，NR增加也常被FR下降抵消。

Push7 后期改为 **HYPIR 教师本体的直接 NR 微调**，没有轻量替代、没有冻结教师输出蒸馏loss。部署对齐模块逐层由FP32 master计算BF16 LoRA delta和BF16 base merge，再做持久前向；旧独立探针记录257层merged权重exact、eager UNet差0、514真实FP32梯度、checkpoint重算。该证明不等于所有后来训练全pipeline都与编译部署数值逐bit相同；实际新模型仍独立生成并评分。

最新分支从A7750重新初始化，freshAdam0，seed20261015，stage100/save50、768原生crop、原点256网格；原HYPIR512tile/stride256、三段Gaussian融合、posterior.sample/DDPM/wavelet不改。配置：LR5e-6、reward.1、MUSIQ1.5、CLIP.1、NIQE权重.5，input L1=.02、LPIPS=.01、gradient=.05、参数anchor=1。NIQE训练使用原实现可微的线性 N/10 surrogate；正式评分仍是原均值/clamp公式，不改评分工具。

为限制结构漂移，**514张量都参与autograd，但Adam/clip只更新6张**，每步清除全514梯度；另508张量在实际里程碑与A逐张量完全相等。参数anchor mean6与RMS口径按选中6张记录，不能伪称全514更新或Adam514。6张共3,113,984元素：

```text
up_blocks.3.resnets.2.conv1.lora_A.default.weight
up_blocks.3.resnets.2.conv1.lora_B.default.weight
up_blocks.3.resnets.2.conv2.lora_A.default.weight
up_blocks.3.resnets.2.conv2.lora_B.default.weight
conv_out.lora_A.default.weight
conv_out.lora_B.default.weight
```

conv_shortcut的A/B没有被选中。新raw仍保存完整514张量；部署不能加载旧A7750的UNet引擎代替新权重。候选manifest SHA `d7e03918355aa9cb459c1d37f8ffafcee89856bb8b23cdf0346b5e440b3046cb`，新UNet engine SHA `6912f4afae77ef4815f8b8d568a9d5eef76ea3c130b554e01987452374ba8a77`。VAE引擎只对应未训练的共享VAE。

## 视觉选择、退回与负结果

1. **高分轻量学生未替代用户认可的HYPIR。** Push5从工具包G4初始化、向A7750测试100伪标签蒸馏250步，原工具预测128.42938；随包G4复现152.66092只是参考，不是新训练成果。学生栏杆/手柄有融化与波纹、FR低于HYPIR，用户查看后明确偏好previous model output，VISUAL_SELECTION记录回到A7750。该G4学生与后来用户指定的包内PixG5i也不能混称同一模型。
2. **39ms channels_last方案没被当成保画质成功。** 报告仅把它列作探索；Push6选择的是无CL版本，经实际103PNG全等才桥接旧质量。时延独立，不能把改变像素的候选速度配旧A质量。
3. **直接全LoRA NR 50/100步出现平滑损细节。** 现有 matched23/98 审计：parent S_subset3.75397915，direct50为3.70570450、direct100为3.59241022。`visuals/hypir_direct_nr_pilot_50_100/OBSERVATIONS.md`明确字边、金属标记、叶脉变软，100更明显，未称保画质提高。
4. **NR-dominant与768不是自动修复。** 对应512的50/100 S_subset3.73825726/3.72261399；768的50/100为3.73454049/3.71725502，均低于同一parent pilot。扩大crop仍需实际指标与视觉门槛。
5. **NIQE全LoRA和参数插值只得到pilot优势，未晋升主模型。** NIQE768 50/100 S_subset3.78912109/3.80308702；α=.25/.5实际FP32参数插值为3.77322228/3.78721732，但FR下降。这些是23图pilot，不是full或65分证据；后来主线改成受限2张/6张更新以控制结构漂移。关于具体固定座改变的历史人工判断需结合相应图库复核，本次不把聊天记忆冒充新图像审阅。
6. **NAF保真网络仅保留准备/探测记录。** `nr_push7_20261015/naf_fidelity_preparation/README.md`是17.1M NAFNet32独立初始化与CPU恒等检查的准备快照，明确没有已验证质量或分数、训练尚未实施；不能把拟定500/1000/2000步计划计成已完成训练，也不属于HYPIR本体改良。其后主线按用户HYPIR教师约束回到同架构优化。
7. **最终6张受限更新只作有限视觉结论。** 7/16/41/83固定ROI保持主要字形、杆连接、花瓣折线和叶脉位置接近A，纹理/边缘/颜色有小变；局部通过不等于所有图无生成伪影。完整分项显示候选NR净增主要靠NIQE贡献约+.021287，CLIP约−.010949、MUSIQ/100约−.003615、MANIQA约+.000646，不能笼统说所有NR指标都更好。

## 环境、端口与缓存重置

本地报告明确记录：早期30207为Ascend910B3；后续30416为4×A100-SXM4-80GB，CUDA实验根目录如上。30416在 `2026-10-15T06:46:00Z` 检测到/cache重置及物理卡更换，随后从持久档案恢复源码/资产/选中权重和TRT运行时，建立新环境。恢复后Torch2.8/cu128、原指标协议与同轮fresh父模型重新评估。**相同端口不等于同一物理实例**，重启前旧分数/速度不作为新比较的直接分母。

早期README中“CUDA提交正在重跑”、A7750首选等句子属于历史快照，不能覆盖08:44的新JSON选择。状态文件也有时间范围：`active_training=false`只描述HYPIR该轮收尾，不等于后续独立PixG5i蒸馏从未启动。

服务器信息只保存任务所需端口、根路径、模型/协议哈希；本文不包含SSH密码、token、私钥或访问凭据。用户之前提到的其他端口及更早round2的23版本清单，不在本次实际读取的HYPIR报告中逐项核实，不补造在线状态。

## 备份和可恢复性边界

旧实验归档/selection/报告保留，不能把全历史大权重和PNG塞入Git。Push4使用精确resume加raw ZIP recipe/gaps恢复A7750；最新分支的checkpoint capsule依赖持久A resume SHA `e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b`。最新selection记录新SFS写入为false，选中引擎/胶囊本地有独立回执；本次只读取选择记录，不再次重建GB级raw或重新验证SFS所有文件。不能把“本地有resume”说成包含所有训练输入/底模/指标和伪标签的自包含归档。

本页与机器表只包含小型文本/哈希。主任务已确认拟合并的私有仓库为 `RedRoman/csig-track2-tinyenhance`，拟归档路径为 `experiments/independent_sr_20261010_20261015/`。仓库操作、分支和大文件策略由主任务处理；本子任务未访问GitHub或写服务器。

## 证据分层与尚未知项

- **本次文档整理直接核对**：本地来源文件的SHA/尺寸；最新selection↔fullaudit↔formaljoin↔visualproof的小文件哈希绑定；表内NR+FR=S数值一致性。
- **既有独立证据**：机器表标记`existing_*`的PASS审计/选择文件。表示旧实验已记录实际PNG/指标/时序验证，不表示此次再次运行了审计或核验大资产。
- **历史报告陈述**：机器表标记`historical_report_statement_not_rerun_here`，保留报告打印精度；不把这类值伪称新鲜重测。
- **未在本子任务证实**：更早round2全部模型清单/最终分数；所有历史分支的大资产目前是否仍可在线读取；当前SSH可用性；全图或独立数据泛化；官方榜分；65或70分HYPIR目标达成（最新记录均未达65）。

## 来源索引与机器表

正文中的路径均相对于本机`score_search/`证据根。完整来源文件清单、字节尺寸、SHA在同目录 `HYPIR_HISTORY_SOURCES.json`；可机读版本/协议分层表为 `HYPIR_HISTORY_VERSIONS.json` 和CSV。该表是本次所选证据的记录集合，含不同批次对照，不宣称项目所有历史版本均已穷尽。

主要报告：`README.md`、`NR_FINETUNE_REPORT_20261014.md`、`NR_PUSH2_REPORT_20261014.md`、`NR_PUSH3_REPORT_20261014.md`、`NR_PUSH4_REPORT_20261014.md`、`NR_PUSH5_REPORT_20261014.md`、`NR_PUSH6_REPORT_20261015.md`、`VISUAL_SELECTION_20261015.md`、`cuda_training/README.md`。

最新状态/选择：`results/current_model_preference.json`、`nr_push7_20261015/results/status_push7_20261015.json`、`nr_push7_20261015/results/nr_push7_lastres_lr5e6_selection.json`。后期逐分支来源为`nr_push7_20261015/evaluation/`下full、matched pilot与speed JSON；机器来源清单逐项列出，不引入备份目录扫描。


### 全部可得报告表与代表图

为避免只保留赢家，本次另逐表提取所读报告/README的 **22 张原始表**（包含训练系数、候选完整/pilot指标、时延与权重哈希），保存到 `HYPIR_HISTORY_REPORTED_TABLES.json`。其中含NR/FR/S的 **51 条显示精度记录**另存 `HYPIR_HISTORY_REPORTED_METRICS.csv`，并嵌入主版本JSON。它们可能与55条结构化记录重复；重复是来源/批次留痕，不相加计算“实验版本总数”。未把缺失全量结果的pilot补造为full。

推荐主任务复制以下三张代表性最终教师对照图。每张均小于4MiB，本次已实际重算本地文件SHA并与冻结visualproof一致；左A7750 fresh full，右最新raw23de fresh full，固定ROI原尺寸1:1。这里只是建议展示，不替代全部图像审阅：

| case / 看点 | score_search相对路径 | 字节 | SHA256 |
|---|---|---:|---|
| 7 / 文字 | `nr_push7_20261015/visuals/hypir_lastres_lr5e6_trt_full_final/case7_hypir_lastres_lr5e6_trt_full_roi_1to1.png` | 817124 | `5ef5b42f4fd30b3df85bb4d313cb8eb83d7cf99dbee9913821c4bfd3c3ba06f6` |
| 16 / 栏杆接头与固定座 | `nr_push7_20261015/visuals/hypir_lastres_lr5e6_trt_full_final/case16_hypir_lastres_lr5e6_trt_full_roi_1to1.png` | 1022660 | `5383655cb7cac57c4859ad94ca51d8a72eee6176dfd89f7e282772135a4f7c3e` |
| 83 / 花瓣与叶脉 | `nr_push7_20261015/visuals/hypir_lastres_lr5e6_trt_full_final/case83_hypir_lastres_lr5e6_trt_full_roi_1to1.png` | 1200327 | `81f1374132712784c75d8224add39c4b9187454d31dfbb89262cc12917db9d7d` |

可机读的三图选择清单：`HYPIR_HISTORY_RECOMMENDED_VISUALS.json`。旧报告原始路径/时间保留，最终GitHub内部链接由主任务按实际目录排布转换。
