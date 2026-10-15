# Push3 同次单卡测速及原工具预测

本阶段比较对象为旧最佳6500版与本轮最终选中权重。两者都使用同一次新G4参考值，重新测parent6500；不会把历史55.767分与新候选测速混合比较。

所有工具均已另存为新源文件，原 `code/benchmark_cuda.py` 与原 `scoring_tool/predict.py` 不修改。

- 原benchmark SHA：`3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995`
- 原predict SHA：`0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1`
- launcher：`code/push3/paired_latency_push3.py`，默认只准备3条argv和可读命令，完全不启动GPU。
- CPU join：`code/push3/join_paired_prediction_push3.py`，不包含任何benchmark启动操作，不导入torch或NumPy。

根代理确认最终selection并使GPU0空闲后，可先查看计划：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python \
  /cache/independent_sr_cuda_20261013/code/push3/paired_latency_push3.py \
  --selection-json /cache/independent_sr_cuda_20261013/results/nr_push3_selection.json \
  --plan-out /cache/independent_sr_cuda_20261013/calibration/push3_paired_latency_plan.json
```

selection尚未写入时，可以改用 `--candidate-label push3_c6750_compiled_all` 等已有full或pilot的render label准备候选测速。该label仅帮助查找真实model_dir和checkpoint，不能代替正式质量评测或最终selection。最终join仍要求完整103/418独立审计PASS，且测速对象的model_dir/checkpoint必须与最终selection精确一致。

只有根代理需要实际调度时，在相同命令上添加 `--execute`。launcher显式设置 `CUDA_VISIBLE_DEVICES=0`，严格按以下顺序启动3个独立新进程：

1. 原G4：`calibration/g4/model_dir` → `calibration/push3_g4_latency.json`
2. 旧最佳6500：原已完整评分的model_dir → `calibration/push3_parent6500_latency.json`
3. 最终候选：实际全量采用的model_dir → `calibration/push3_selected_latency.json`

三个进程均使用原benchmark `--device 0 --warm 30 --n 200 --seed 231`，输入为512×512 CUDA FP16，保留200个event与wall原始样本。parent/candidate显式 `--hash-file` 绑定各自原始权重，并把第1次新G4的event p10作为 `--ref-ms`。所有模型均在同一卡上顺序执行，初始化、编译、warmup、hash均不计入推理时间。

launcher在每次进程前后使用NVML查询物理GPU0的UUID/PCI bus/name，并确认每次启动前GPU0无其他计算进程。它将实际命令、环境选择、时间戳、物理卡身份和每份原始测速输出SHA写入 `calibration/push3_latency_launch.json`。为保持新鲜标定证据，若固定输出文件或既有launch记录已存在，实际执行会拒绝覆盖；根代理应保留并审查已有证据。

完成实际3次测速及正式selection后，只运行CPU join：

```sh
/cache/independent_sr_cuda_20261013/env/bin/python \
  /cache/independent_sr_cuda_20261013/code/push3/join_paired_prediction_push3.py \
  --selection-json /cache/independent_sr_cuda_20261013/results/nr_push3_selection.json \
  --out /cache/independent_sr_cuda_20261013/calibration/push3_paired_prediction.json
```

join独立执行以下核验：

- 新候选与旧parent均为100张test＋3张validation的独立审计PASS，并核验103张实际PNG、输入、render/scored记录、源代码、原评分权重和checkpoint的实际SHA。
- 重新核验render protocol身份、实际model_dir配置与源；逐图重算418个标量均值、NR、FR和S，拒绝pilot或不完整结果。
- 校验3份benchmark原源SHA、实际artifact文件和外部checkpoint，确保质量与速度属于同一权重、同一推理配置/源文件。
- 按200个原始样本重新计算event/wall p10、中位数和最小值，核验warm30/n200/seed231/512 FP16、有效GPU输出、同硬件runtime及同次G4换算。
- 校验3次物理UUID与PCI bus全部一致、执行顺序和固定CUDA0、launch保存的输出SHA。
- 最后导入SHA锁定的原predict.py调用原函数；不复刻或修改其公式。

输出分别列出本次parent预测、新候选本次测速预测，以及固定本次parent速度时仅由质量变化产生的预测。`quality_factor_vs_parent6500` 与 `measured_timing_factor_not_proven_algorithm_speedup` 分开记录，后者只是这一次的计时描述。0.3ms等细小差别不证明算法稳定加速；工具经验预测亦不是官方实测榜分。

准备过程只做CPU语法/解析及计划验证，子代理没有执行 `--execute`。源码SHA和只读试验记录见 `paired_latency_preparation_static_audit.json`。最终calibration/push3*小记录会进入selected-only备份；无需复制原基础模型或重复原评分权重。
