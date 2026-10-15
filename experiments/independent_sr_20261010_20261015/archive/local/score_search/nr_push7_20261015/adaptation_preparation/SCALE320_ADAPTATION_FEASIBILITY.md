# Scale320 后备适配：CPU可行性与静态协议审查

本轮主线为A7750相同权重/架构的统一缩放残差推理，先等scale.75与.625实际pilot和速度。这里没有新训练源、训练启动或GPU调用；旧源码、权重、结果、归档均不改。coeff170/180准备已按root要求停止。用户拒绝的G4学生不作为此路线。

## 已有推理实现的独立检查

已只读审查 `R/code/push7/scaled_residual_runner.py`，SHA `063e228fdbeade003b71506494fd9097ec8c6c766f95e7e1a2d5a6d20cc5b721`。两实际配置为 `R/submissions/hypir_push7_a7750_s075_residual/model_dir` 与 `...s0625_residual/model_dir`。

512与native都按同一个公式运行：`hs=max(8,8*round(H*s/8))`、`ws=max(8,8*round(W*s/8))`，Python ties-to-even；FP32 bicubic/align_corners=False/antialias=True缩小；原HYPIR完整小图enhance；`native_lq + up(enhanced_small-small_lq)`，最后clamp到[0,1]、转换[-1,1]。小图无额外clip，无channels_last，无按case/512识别分支。s=.75对应patch384/stride192；s=.625对应320/160，保持原生视野约512/256和50% overlap。A7750 raw固定，model_t200/coeff_t175固定，初始化前seed231且不中途reset。编译仅既有tile functions/wavelet。

静态主链PASS；实际有限输出、ownership、连续RNG和速度需root实际运行验证。“保留HF”只指保留原生输入的高频基底，不保证与A7750输出高频完全相同。下采样可能保留输入噪声/模糊，改变修复能力；所有输出像素及随机消耗变化均已在protocol声明。必须新103/418，不继承A质量。

## 65分的S预算（条件表，不是scale实测分）

采用最后一批G4 p10=.6492159962654114ms、A S=3.7614565663241937与未改原predict公式。最终必须重新同卡G4→A→新候选fresh配对。本表只用于判断实测速度是否给短适配留有质量预算；pilot的S_subset不能直接代入正式分。

|候选实测p10(ms)|达到65所需完整S|相对A允许最大S降幅|
|---:|---:|---:|
|10.000|3.526678|+0.234779|
|12.000|3.577699|+0.183758|
|15.000|3.640489|+0.120967|
|18.000|3.692028|+0.069429|
|20.000|3.721893|+0.039564|
|22.000|3.748955|+0.012502|
|22.989|3.761457|+0.000000|
|25.000|3.785312|-0.023855|

同41.438ms时要65需S=3.929578（+.168121）；只做小步NR通常不够。原500步完整S只增加.001762，不能据此承诺短训补大缺口。若scale320速度约20ms，则最多容许S下降.03956；18ms约.06943；22ms仅.01250。负降幅代表反而需要提升S。

## 只有真实pilot表现接近可接受时，考虑一个短适配分支

建议先比较两scale的实际速度、同RNG/同native上下文四例（7中文字、16栏杆、41与83花瓣）以及原23图S_subset。不要对明显文字/直线融化的候选靠训练赌修复。只对速度确实有预算、视觉结构仍可靠的scale320考虑适配；距65的质量缺口若明显大于旧短训收益，不盲目延长训练。pilot用于筛选，最后仍完整103/418才有工具预测。

初始化为封存A7750 exact FP32 LoRA514，取immutable resume_step7750.pt中的lora逐张复制，raw589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d与resume e240f3e810654e9503ed620a1007649d2dd056c290960b442e220e0a1451a83b精确一致。VAE、text encoder与base冻结；只适配原LoRA，不改架构/评分网络。新任务已经改变分辨率分布，建议显式新Adam0而非默默继承旧NR一阶矩；需root确认实现决策，不能写成exact Adam续训。起步LR建议1e-7、先25步停审、最多125步，不自动扩张多分支。新RNG独立记录；如果root要求继承父4RNG/Adam，必须如实记录继承和首步旧moment漂移，不能声称自锚即零漂移。

teacher仅为现成A7750原生100张test PNG：`R/nr_push4_full_evaluations/push4_a7750_compiled_all/render/test/case{i}.png`。manifest SHA35355c94cace390b5f5918cfb607d85b0f55c9cd1e1ba0a4f6a4d034abb54e16；render_protocol SHA c0c2389b844b83ccda971f9179b04221f0a3751add428deef3bfc96ba80a0dc7。正式训练前再次逐PNG实际SHA/尺寸/对应原JPEG SHA核对，只纳入test case1..100，排除val LQ和GT。无需在线teacher模型，也不重新生成伪标签。

令D/U精确使用部署FP32 bicubicAA及尺寸规则，x为官方FP16往返后的native输入，T为native A PNG。可表达的teacher应投影到残差分支：`small_target=clamp(D(T),0,1)`，`residual_target=small_target-D(x)`，`projected_teacher=clamp(x+U(residual_target),0,1)`。这保留相同native输入HF基底，同时只教小图分支可表达的低频修复。直接逼320输出拟合完整A的所有原生高频，容易把不可表达的细边变成晕圈/纹理，不建议。

主要loss可用`2*L1(native_student,projected_teacher)+.2*LPIPS(...)+.6*gradient_L1(...)`；仅弱NR奖励（例如.01×[1.5*MUSIQ/100+MANIQA+.1*CLIPIQA]，不是提升NR权重）作用于部署重建后的native中心512。指标保持原权重且冻结，MANIQA训练20/正式100。强目标在新分辨率下初始fidelity应非零，不应沿用自教师零loss断言。先核三真NR输入梯度与514份LoRA有限梯度，后实际更新并保存新raw/Adam/RNG/计数。

训练窗口建议native1024上下文→small640，严格同320/160 Gaussian tiler与融合，最后只在native中心512计算loss。crop原点对齐256原生tile网格（同时满足5/8缩放相位），避免crop后resize/overlap相位漂移；尺寸/插值/原JPEG FP16往返与部署一致。不能拿普通512→320单tile训练就宣称等价于完整native拼接。9个small tile的反传增加训练成本，可用原LoRA gradient checkpointing；具体显存/耗时须由root实际预检。原enhance/tiled推理有no_grad路径，实施时需要NEW可反传同算术入口，不能直接调用后再声称梯度到达，也不能改旧原函数或省略融合。

任何更新都新目录、显式source/parent/teacher/input catalog；25步与最终125保存不可变raw+matching resume+实际SHA。正式评估用新部署runner/source与新权重，完整103/418与fresh配对；训练crop分数绝不当正式提升。若首25步固定文字/直线边界变差或NR/FR明显退，不继续到125，保留未训练scale候选或A。该方案是候选可行性，不是已实现/已运行成绩。

## 原full scorer命令（root调度执行）

两个pilot的label/目录仍由root正式指定；如下只展示真实scorer接口。先完成原renderer的`--cases all`，任何scaled PNG变化都必须全103图重评。只对root最后选择的一条做完整验收，不自动跑两条full。

```sh
RUN_ROOT=/cache/independent_sr_cuda_20261013
RUN_PY="$RUN_ROOT/env/bin/python"
RUN_LABEL='<root-selected-full-label>'
RUN_FULL="$RUN_ROOT/nr_push7_full_evaluations/$RUN_LABEL"
CUDA_VISIBLE_DEVICES=0 "$RUN_PY" "$RUN_ROOT/code/score_cuda_stream.py" --tool-dir "$RUN_ROOT/scoring_tool" --evaluation "$RUN_LABEL=$RUN_FULL/render" --out-root "$RUN_ROOT/nr_push7_full_evaluations"
```

`--evaluation`必须传full/render，默认test100/val3/MANIQA100、split seed1234连续，不用pilot cases参数。其独立CPU审计应核actual PNG SHA与评分行、native尺寸、原source/input/GT/weights、418均值原公式。最终展示只拉所选实际full的四例原PNG与同坐标1:1 ROI，不使用pilot PNG冒充full。
