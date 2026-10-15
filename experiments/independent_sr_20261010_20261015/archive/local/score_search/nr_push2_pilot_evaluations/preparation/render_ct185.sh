#!/usr/bin/env bash
set -euo pipefail
# Prepared only. Parent scheduler must select an idle GPU and execute explicitly.
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES before running}"
/cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/render_cuda_submission.py --model-dir /cache/independent_sr_cuda_20261013/submissions/hypir_cuda_nr_s6250_ct185_compiled/model_dir --test-dir /cache/independent_sr_cuda_20261013/source_payload/assets/input --val-dir '/cache/independent_sr_cuda_20261013/source_payload/scoring_assets/data/验证集' --out-dir /cache/independent_sr_cuda_20261013/nr_push2_pilot_evaluations/nr_s6250_ct185_compiled_pilot --cases pilot
