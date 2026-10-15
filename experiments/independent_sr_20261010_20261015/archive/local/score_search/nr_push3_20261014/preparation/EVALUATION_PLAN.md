# 第三轮评测准备（CPU 检查，GPU 调度由主代理负责）

父版本为完整评分的 6500 步 M1.5，原权重 SHA256 为 `cfcc3013fa87aa721f346d6cc285e70b6d6ff2c88315ce9f1465b07ffe020e68`。
本轮保留评分工具、指标权重、MANIQA 100 crops、推理种子 231 和各评分 split 起始种子 1234，训练目标变化不改变评分公式。

候选完成并得到实际权重 SHA256 后：

```sh
R=/cache/independent_sr_cuda_20261013
PY=$R/env/bin/python
$PY $R/code/push3/register_push3_candidate.py --label push3_a6750_compiled --checkpoint "$R/adaptation/runs/实际目录/step6750.pth" --checkpoint-sha256 实际SHA256 --execute
```

该命令创建新的 submission，禁止覆盖既有目录；仅允许 label、weight_path 以及明确指定的 coeff_t 变化。默认 model_t=200、coeff_t=175。runner 原样复制。返回的 JSON 包含 pilot/full 的冻结 renderer 与原评分代码命令。评分必须使用新目录和同一 registration label。

多候选命令生成（无 GPU 启动）：

```sh
$PY $R/code/push3/make_push3_eval_commands.py --label push3_a6750_compiled --label push3_b6750_compiled --scope pilot --write-dir "$R/nr_push3_pilot_evaluations/preparation/6750_jobs"
```

`render_sequential.sh` 是多候选顺序推理脚本，`score_contexts.sh` 使用原评分器独立 RNG contexts。主代理可安排不同 GPU、同卡顺序或待 PNG 到达再评分；不得同进程混用 candidate 的 RNG 状态。

Pilot 固定 test20 + validation3，即 23 张、98 个标量，结果只称 `S_subset`。测试 case 为 1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100。原 scorer 对 pilot 发布的 `S_proxy` 字段不等于完整分数。
完整为 test100 + validation3，即 103 张、418 个标量，称 `S_proxy = NR(test100)+FR(val3)`。不得将部分到达的均值或 pilot 分数写为完整分数。

完成后的 CPU 审计：

```sh
CUDA_VISIBLE_DEVICES='' $PY $R/code/push3/audit_push3_evaluation_cpu.py push3_a6750_compiled_pilot --evals-root "$R/nr_push3_pilot_evaluations"
```

审计会检查实际 PNG/尺寸/RGB/输入/GT/逐图指标 SHA，重算均值和公式，核对原指标权重、base assets、runner/repo/render/scorer source hashes、各步骤 RNG checkpoint，并与同 scope 的旧 6500 M1.5 进行正式配置与最终 RNG 比较。所有 torch load 都 map_location=cpu，不初始化 CUDA。注册记录属于新增 submission 外层，避免修改 render 的 model_dir identity。

最终全量胜者选定后，再取 case7、16、41、83 的新完整输出，与此前保留的 6500 M1.5 原图作固定 ROI 对比。下载前后核对完整 manifest SHA，不对原图加锐化/调色。按主代理空间策略，仅保留新胜者 full103、精确续训权重与 metadata 的持久备份；本轮无需重新下载所有 pilot 原图。
