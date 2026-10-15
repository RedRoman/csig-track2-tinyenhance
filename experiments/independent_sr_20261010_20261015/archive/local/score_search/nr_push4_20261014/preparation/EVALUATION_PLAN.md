# Push4 评测准备（2026-10-14，尚未启动候选评测）

恢复与评分对照均为 Push3 C7250：raw SHA256 `fd733144c370352f460097037c8cb7cfd45226d3f0bbdf9e8274846ee9ed5a8e`，匹配 immutable resume SHA256 `b8b709f284ce627113e392ede00e43bcd522c8dc23b63719e9fcbaa573c7256d`。本轮训练的保真 teacher 另为6500，不能与评测对照混淆。

根代理已确定三条各500步计划，保存 local250/global7500/Adam1500 和 local500/global7750/Adam1750：

| 建议label分支 | 实际run目录名 | MUSIQ训练系数 | NR总reward |
|---|---|---:|---:|
| a | cuda_nr_stage5_s7250_teacher6500_continue500 | 1.5 | 0.15 |
| b | cuda_nr_stage5_s7250_teacher6500_m25_500 | 2.5 | 0.15 |
| c | cuda_nr_stage5_s7250_teacher6500_m25_r10_500 | 2.5 | 0.10 |

三分支计划均 LR2e-7、CLIPIQA训练系数0.5、spatial-gradient权重0.3。这些属于训练计划，必须另以根代理提供的实际v5源、CLI配置、启动protocol、完成marker及独立训练审计核验；本评测准备不捏造已完成的训练证明。

## 冻结来源与本轮最小改动

旧Push3脚本、报告、审计和结果均不改动。新源部署在 `/cache/independent_sr_cuda_20261013/code/push4/`，本地对应 `nr_push4_20261014/preparation/`。

- `register_push4_candidate.py`：父配置改成实际C7250 model_dir；以其完整评测protocol核验原render/scorer/env/tool/repo源。原runner SHA `396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae`。仅修改新label、实际新checkpoint path及明确指定的coeff_t（默认175）。保持 model_t200、seed231、BF16、FP16输入、round PNG、缓存/融合/compile/tile协议。只接受上表三run的现存step7500或7750及调用方给定的实际SHA；禁止覆盖旧submission。没有训练权重时不注册候选。
- `make_push4_eval_commands.py`：仅替换本轮输出根和registration schema。生成独立render命令及原scorer的多label独立RNG contexts命令，不启动GPU。
- `audit_push4_evaluation_cpu.py`：复用Push3的完整核验逻辑，仅把默认reference改成旧C7250同scope，新增C7250 raw SHA guard；仍验证23/98或103/418、实际PNG/输入/GT、source、base、指标权重、统计公式与最终四类RNG。不会要求物理卡编号固定；CUDA_VISIBLE_DEVICES映射后的logical0和相同A100均正常审计。

新审计器的代码主体不要求stage4或stage5路径；它以实际render protocol/runner/checkpoint SHA为准。训练v5源/配置及精确Adam/RNG迁移由独立训练审计覆盖，之后最终selection再同时绑定两类审计。

## 候选实际完成后使用

```sh
R=/cache/independent_sr_cuda_20261013
PY=$R/env/bin/python
$PY $R/code/push4/register_push4_candidate.py \
  --label push4_a7500_compiled \
  --checkpoint "$R/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7500.pth" \
  --checkpoint-sha256 实际已封存SHA256 --execute
```

注册默认只dry-run；`--execute`才创建新的 `submissions/hypir_push4_a7500_compiled/model_dir`。registration.json保存在model_dir外层，防止改变render identity。

```sh
$PY $R/code/push4/make_push4_eval_commands.py \
  --label push4_a7500_compiled --label push4_b7500_compiled --label push4_c7500_compiled \
  --scope pilot --write-dir "$R/nr_push4_20261014/preparation/7500_eval_jobs"
```

原renderer继续 `R/code/render_cuda_submission.py`，原scorer继续 `R/code/score_cuda_stream.py`。根代理单独设置GPU并启动；准备助手不运行GPU、不启动无限watcher。

## 评分口径及完成审计

Pilot目录 `nr_push4_pilot_evaluations/`，固定20测试case 1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100 加3验证，23张/98标量，只称 `S_subset`。原scorer需明确传相同 `--test-cases`；不能把其原JSON名为S_proxy的pilot字段当完整分数。

Full目录 `nr_push4_full_evaluations/`，完整100测试加3验证，103张/418标量，称 `S_proxy=NR(test100)+FR(val3)`。最多两组full由根代理在pilot完成后决定，不能事先虚构结果。

```sh
CUDA_VISIBLE_DEVICES='' $PY $R/code/push4/audit_push4_evaluation_cpu.py \
  push4_a7500_compiled_pilot --evals-root "$R/nr_push4_pilot_evaluations"
```

默认reference分别是 `nr_push3_pilot_evaluations/push3_c7250_compiled_pilot` 和 `nr_push3_full_evaluations/push3_c7250_compiled_all`。禁止混用pilot/full编号PNG、case顺序和RNG状态。正式MANIQA100 crops、NR FP32、FR FP16 autocast、原metric/network权重与公式不改变；训练MUSIQ reward系数不能写成正式评分系数。

证明存放新目录 `R/audits/push4/evaluations/`。本准备只CPU re-audit已有C7250同scope基准，实际新候选的marker、raw SHA和训练审计由根代理完成后通知注册与GPU排程。
