#!/usr/bin/env bash
set -euo pipefail
# Parent scheduler must verify stage100 checkpoint state and real weight SHA before launch.
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES before running}"
test -s /cache/independent_sr_cuda_20261013/adaptation/runs/cuda_nr_stage2_s6250_cont250/step6350.pth
/cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/render_cuda_submission.py --model-dir /cache/independent_sr_cuda_20261013/submissions/hypir_cuda_nr_stage2_cont_s6350_compiled/model_dir --test-dir /cache/independent_sr_cuda_20261013/source_payload/assets/input --val-dir '/cache/independent_sr_cuda_20261013/source_payload/scoring_assets/data/验证集' --out-dir /cache/independent_sr_cuda_20261013/nr_push2_pilot_evaluations/nr_stage2_cont_s6350_compiled_pilot --cases pilot
