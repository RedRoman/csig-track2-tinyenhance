#!/usr/bin/env bash
set -euo pipefail
# Prepared only. Parent scheduler must verify training complete and weight SHA before launch.
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES before running}"
test -s /cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage2_s6250_roundste250/step6500.pth
/cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/render_cuda_submission.py --model-dir /cache/independent_sr_cuda_20261013/submissions/hypir_cuda_nr_stage2_roundste_s6500_compiled/model_dir --test-dir /cache/independent_sr_cuda_20261013/source_payload/assets/input --val-dir '/cache/independent_sr_cuda_20261013/source_payload/scoring_assets/data/验证集' --out-dir /cache/independent_sr_cuda_20261013/nr_push2_pilot_evaluations/nr_stage2_roundste_s6500_compiled_pilot --cases pilot
