# Push6：最小正式验收协议（CPU审阅，2026-10-15）

本文件只读核对已安装脚本及 `--help`（四者均exit0），没有执行GPU推理、渲染、评分或测速。root现已将待正式验收的唯一候选收敛为无channels_last转换的RO+wavelet组合，不额外扩分支；最终是否选中仍待质量和测速证据。所有旧源、A7750结果和权重保持冻结。

## 核对的原入口

|入口|实际SHA256|原参数/输出|
|---|---|---|
|`R/code/benchmark_cuda.py`|`3e85026bd91ad8fb7fb91c5d3c74211cc33a9cff0f8316f7e56cde864ae41995`|`--model-dir --out --ref-ms --n --warm --device --seed --hash-file`；`--hash-file`可重复；直接输出完整JSON|
|`R/scoring_tool/predict.py`|`0af39f578f704e00a0a0a5ce55e58f1668d283ac4602065d8215b6d21cf433e1`|`--nr --fr --ms --ref_ms --label`；注意此处ref参数是下划线；没有JSON选项，stdout预测只保留一位小数|
|`R/code/render_cuda_submission.py`|`1b578c945ba4b33c7d6ce522151bb9c3a8d3a004eab0c5c7e802fad6ae1bb1f5`|`--model-dir --test-dir --val-dir --out-dir --cases all`|
|`R/code/score_cuda_stream.py`|`20125b94596c669f33dc99b592424a8c1e6a44ec1247d304853823cc5e399074`|`--tool-dir --evaluation label=render目录 --out-root`；默认test100，正式MANIQA100|

`R=/cache/independent_sr_cuda_20261013`。A7750实际raw SHA为`589324563e9e223291d6b58ab8a02d38614582279323bdd20d19460d0d73977d`，其既有full目录为`R/nr_push4_full_evaluations/push4_a7750_compiled_all`。不能因权重相同直接复用旧图/分数；runner数值路径改变仍需新验证。

## 1. 质量路径：先完成一次原生103图，再判复用或全量评分

沿用已批准的新runner/model_dir和实际label，先核对它的runner/config/source SHA与已测试候选一致、raw仍为上述A7750。已有512连续多输入/输出ownership/四RNG和四例原生1:1红线检查不能由测速替代。正式渲染必须fresh process、无测速warmup；runner在初始化前seed231，按val1..3→test1..100连续RNG。不是逐图重置seed，不拿benchmark热身后的状态渲染。

本轮已计划正式label为`push6_a7750_ro_wavelet_all`、model_dir为`R/submissions/hypir_push6_a7750_ro_wavelet/model_dir`；下述full目录仍为planned验收目录，不表述为已完成。远端实际runner SHA为`dfd91811af6fa1581f246e19a04f8ada10515864ca9f7d3a35fbcc5e504f9f9b`，config SHA为`8ea2367f3395cb87e98e663fb47c172de97632913776e9514a8ac44247b31cd8`。RUN_FULL必须是新的实际full目录（或其同协议精确resume），不覆盖A目录。

```sh
set -euo pipefail
RUN_ROOT=/cache/independent_sr_cuda_20261013
RUN_PY="$RUN_ROOT/env/bin/python"
RUN_LABEL=push6_a7750_ro_wavelet_all
RUN_NEW="$RUN_ROOT/submissions/hypir_push6_a7750_ro_wavelet/model_dir"
RUN_FULL="$RUN_ROOT/nr_push6_full_evaluations/$RUN_LABEL"

CUDA_VISIBLE_DEVICES=0 "$RUN_PY" "$RUN_ROOT/code/render_cuda_submission.py" \
  --model-dir "$RUN_NEW" \
  --test-dir "$RUN_ROOT/source_payload/assets/input" \
  --val-dir "$RUN_ROOT/source_payload/scoring_assets/data/验证集" \
  --out-dir "$RUN_FULL" --cases all
```

渲染完成需有`render_complete`、complete状态、manifest/protocol完整103条，唯一顺序val3+test100，actual PNG SHA/原生RGB尺寸与manifest及输入匹配，来源/权重/source协议可核。

- **全部103张actual PNG字节SHA与A7750逐一相等**：并且同103输入、同3GT、同原评分协议及权重，才允许复用A的已审计418个指标与NR/FR。新建小的exact103对照证明，绑定新/旧manifest SHA、每张实际SHA、输入与原评分protocol；另外核对新/旧render_progress.pt终点Python/NumPy/Torch CPU/CUDA四RNG精确相同，明确该证据只描述终点，不宣称逐图RNG；明确写quality_reused_from=A，不能伪造一份新scorer结果或仅比较manifest声明。
- **任一PNG SHA不同，或证据不完整**：整个103张重新评分，不只评分变化图/四例/pilot。不用浮点误差小或肉眼相似代替正式质量评估。

```sh
CUDA_VISIBLE_DEVICES=0 "$RUN_PY" "$RUN_ROOT/code/score_cuda_stream.py" \
  --tool-dir "$RUN_ROOT/scoring_tool" \
  --evaluation "$RUN_LABEL=$RUN_FULL/render" \
  --out-root "$RUN_ROOT/nr_push6_full_evaluations"
```

关键：`--evaluation`传 **full目录/render**，不是full根目录；脚本会继续拼val/test。正式结果应有scores/result.json和protocol.json、measurements.jsonl，complete103、official_full_count=true、scope=full100_test_NR_plus3val_FR，val3×6+test100×4=418个有限指标。独立CPU核对actual103 PNG SHA=manifest=评分行sr_sha256，重算逐项均值后再计算：

`NR = mean_test(CLIPIQA)+mean_test(MANIQA)+mean_test(MUSIQ)/100+max(0,(10-mean_test(NIQE))/10)`

`FR = 2-mean_val(LPIPS)-mean_val(DISTS)`；`S=NR+FR`。先均值再NIQE clamp，不能平均逐图S。原指标/权重不变、MANIQA100、评分seed1234按每split连续抽样。固定四例1:1只作视觉红线，任何文字笔画/栏杆结构退化仍拒绝；验证GT只作正式评测。

## 2. 正式速度：同一空闲物理GPU0，G4→A→new三个串行fresh process

在所有渲染/评分/其他本轮GPU任务结束后测。如root已启动同批正式三连，以那一批完整结果为准，不因本流程重复测速。每进程只加载一个Runner，外部`CUDA_VISIBLE_DEVICES=0`和逻辑`--device 0`保持一致；每个fresh process seed231、随机CUDA FP16[1,3,512,512]输入、warm30、n200。benchmark在构造、hash、输入分配、30次warmup后才记录200次完整`Runner.infer` CUDA event与wall样本；正式预测只用event_p10（numpy默认linear percentile）。不能用init_s、wall_min、event_min或未热身首call替换。

```sh
RUN_CAL="$RUN_ROOT/calibration/push6_acceptance_$RUN_LABEL"
mkdir "$RUN_CAL"
RUN_G4="$RUN_ROOT/calibration/g4/model_dir"
RUN_A="$RUN_ROOT/submissions/hypir_push4_a7750_compiled/model_dir"
RUN_RAW="$RUN_ROOT/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/step7750.pth"
RUN_CONFIG="$RUN_ROOT/adaptation/runs/cuda_nr_stage5_s7250_teacher6500_continue500/config.json"

nvidia-smi -i 0 --query-gpu=index,uuid,pci.bus_id,name --format=csv,noheader > "$RUN_CAL/gpu_before.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader > "$RUN_CAL/processes_before.csv"

CUDA_VISIBLE_DEVICES=0 "$RUN_PY" "$RUN_ROOT/code/benchmark_cuda.py" \
  --model-dir "$RUN_G4" --out "$RUN_CAL/g4_latency.json" \
  --device 0 --seed 231 --warm 30 --n 200 \
  --hash-file "$RUN_G4/model.pth"

RUN_REF_MS=$("$RUN_PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"]=="complete"; print(x["latency_512"]["event_p10"])' "$RUN_CAL/g4_latency.json")

CUDA_VISIBLE_DEVICES=0 "$RUN_PY" "$RUN_ROOT/code/benchmark_cuda.py" \
  --model-dir "$RUN_A" --out "$RUN_CAL/a7750_latency.json" \
  --device 0 --seed 231 --warm 30 --n 200 --ref-ms "$RUN_REF_MS" \
  --hash-file "$RUN_RAW" --hash-file "$RUN_CONFIG"

CUDA_VISIBLE_DEVICES=0 "$RUN_PY" "$RUN_ROOT/code/benchmark_cuda.py" \
  --model-dir "$RUN_NEW" --out "$RUN_CAL/new_latency.json" \
  --device 0 --seed 231 --warm 30 --n 200 --ref-ms "$RUN_REF_MS" \
  --hash-file "$RUN_RAW" --hash-file "$RUN_CONFIG"

nvidia-smi -i 0 --query-gpu=index,uuid,pci.bus_id,name --format=csv,noheader > "$RUN_CAL/gpu_after.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv,noheader > "$RUN_CAL/processes_after.csv"
```

顺序是三个独立命令，不能并行；同权重并不允许跳过A fresh对照。G4与A/new都要成功exit0，原benchmark实际SHA一致，三份JSON的seed231/input512FP16/n200/warm30一致，200条event/wall样本均有限正数，重算p10与摘要吻合，输出CUDA原尺寸有限且范围[-1,1]合规（benchmark只报告范围，不能只看status=complete）。新runner/ckpt实际SHA必须匹配质量路径，baseline A runner SHA仍为`396e29afc3b60e077a8ef200cb4c382b33abac4fb60ab37dd9a873eb369704ae`。

benchmark的`--out`已包含模型目录源/配置/权重清单、extra_files实际SHA、原benchmark源SHA、Torch/CUDA/cuDNN/GPU型号、输入输出、计时范围、原始200样本、p10/median/min/wall统计。它**没有**记录物理UUID/PCI/CUDA_VISIBLE_DEVICES，也不遍历符号链接目录；launch sidecar需保存以上物理卡前后记录、明确CVD=0和执行顺序/exit0；外部ckpt已用重复`--hash-file`绑定，外部HYPIR/base源还要与正式render protocol中既有SHA核对。不能只用相同GPU型号当作同一物理卡证据。

## 3. 原公式：用本轮G4原始p10归一化，保存未舍入JSON

原predict参数固定`a=.49,k=2.17,anchor={score:152.95,S:4.0841,g:.431}`。令本轮G4原始event_p10为r、候选原始event_p10为t：

`g=t*.431/r`

`prediction=152.95*2**(S-4.0841)*((.49+2.17*.431)/(.49+2.17*g))**.2`

不混入历史55.7/56.4/58.7或历史G4毫秒。A与new用同一个本轮r。向原CLI传**原始t**与`--ref_ms r`；不要把benchmark已归一化的g_used_ms再传`--ref_ms`而重复归一化。

```sh
# RUN_NR/RUN_FR来自本轮103/418独立审计；仅exact103证明通过时取A已审计值。
# RUN_NEW_MS取本轮new_latency.json的latency_512.event_p10。
"$RUN_PY" "$RUN_ROOT/scoring_tool/predict.py" \
  --nr "$RUN_NR" --fr "$RUN_FR" --ms "$RUN_NEW_MS" \
  --ref_ms "$RUN_REF_MS" --label "$RUN_LABEL"
```

本轮CPU join应新建Push6副本，不能直接套Push5学生join（其绑定Pix100参数、G4初始化和reference30000）。旧Push4 auditor的runner metadata差异白名单可能拒绝新后端来源；新比较需如实记录该差异并锁定原输入/GT/权重/指标，不能改旧审核源或隐藏后端变化。

原predict仅输出文本，不能加不存在的`--json/--out`参数。CPU join须先核SHA，再用`importlib`直接加载这份未改predict.py并调用`module.predict(S,g)`保存未舍入float；不要解析其一位小数stdout为最终成绩，也不改公式。最终新JSON最少绑定：三份latency actual SHA与raw200复算结果、UUID/PCI/顺序、benchmark/predict actual SHA、新runner/ckpt/source SHA、实际质量来源与103/418 audit或exact103 reuse proof SHA、NR/FR/S、raw_event_p10、reference_event_p10、normalized_g_ms、完整prediction、target70及reaches_target。应同时计算本轮fresh A预测作基线，并保留质量因子`2**(S_new-S_A)`与速度因子`((.49+2.17*g_A)/(.49+2.17*g_new))**.2`，两者相乘恰为预测比；全103 exact时质量因子应为1。只按原经验工具预测表述，不冒称官方已提交榜分。

## 最小通过条件

同一候选的来源链和native103完整；changed则新103/418独立PASS，exact则真实103逐字节对照PASS；四例质量红线通过；同物理GPU0三fresh计时完整；原公式未舍入CPU join与所有SHA绑定。达到70且质量合格即停止扩张。未达目标或出现结构退化，保留A7750，不以学生128分替代用户已认可质量。
