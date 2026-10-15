# Push4 新鲜同卡配对测速准备

新助手 `code/push4/paired_latency_push4.py` 默认只生成计划。只有根代理在正式 full103 选型完成、物理 GPU0 空闲后显式 `--execute` 才启动测量；`--candidate-label` 只可准备 dry-run，禁止拿部分/pilot候选启动正式配对。当前准备没有任何实际Push4测速或预测。

固定顺序为 G4 → 父C7250 → 正式选中模型；每个独立新进程均GPU0、warm30、n200、seed231、512×512 FP16，原 `code/benchmark_cuda.py` SHA `3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995` 不改动。启动前后读取物理UUID/PCI并检查GPU0空闲；每个stage之前也检查，防止串入其他计算进程。

父来源是原冻结 `results/nr_push3_selection.json`，raw SHA `fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e`，full参考审计采用本轮新CPU证明 `audits/push4/evaluations/parent7250_full_cpu_audit.json`。旧selection绑定的原audit SHA与新re-audit SHA不同，因此新join只在内存副本中换成新audit并分别记录两份provenance，不写旧文件。

正式选型后先dry-run：

```sh
R=/cache/independent_sr_cuda_20261013
PY=$R/env/bin/python
$PY $R/code/push4/paired_latency_push4.py \
  --selection-json "$R/results/nr_push4_selection.json" \
  --plan-out "$R/nr_push4_20261014/preparation/paired_latency_plan.json"
```

根代理确认GPU0空闲后，原命令加 `--execute`。如果有旧的Push4 launch或测速输出即拒绝覆盖。实际固定输出：`calibration/push4_g4_latency.json`、`push4_parent7250_latency.json`、`push4_selected_latency.json`、`push4_latency_launch.json`。本轮不会重用Push3的G4/parent/selected计时。

actual warm30/n200三次成功并有正式full selection后，只运行CPU join：

```sh
CUDA_VISIBLE_DEVICES='' $PY $R/code/push4/join_paired_prediction_push4.py \
  --selection-json "$R/results/nr_push4_selection.json" \
  --out "$R/calibration/push4_paired_prediction.json"
```

CPU join要求父与候选均full103/418独立审计PASS，重读实际PNG/原图/逐图指标/原metric权重/source，核验benchmark model_dir/config/checkpoint与质量评测完全对应，并从200份event/wall样本重算p10/median/min。只调用原 `predict.py`（SHA `0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1`）固定公式。

输出单列同次父/新模型预测、固定父速度的quality-only预测、`2**ΔS`质量因子与计时因子。一次很小的计时差不能解释为算法稳定加速；公式只是原五点经验工具估计，不能称为官方榜单成绩。

CPU smoke `smoke_paired_push4_cpu.py` 会硬阻断GPU subprocess，使用父==父只测试真实来源的dry-run计划、200样本参数与新parent audit的完整103源绑定；不会写入任何 `calibration/push4_*`。新join的 `full_quality`、`percentile`、`timing` 三个核心函数与冻结Push3逐字节相同。另可对实际已生成render_protocol的 `--candidate-label push4_c7500_compiled_pilot` 做只读计划；其不是完成的Push4质量或速度结果。
