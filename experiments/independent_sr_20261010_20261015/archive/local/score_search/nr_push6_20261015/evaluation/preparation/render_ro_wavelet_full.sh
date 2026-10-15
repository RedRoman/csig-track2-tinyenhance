#!/usr/bin/env bash
# PREPARATION ONLY. Root alone assigns PUSH6_GPU and starts the frozen renderer.
set -euo pipefail
: "${PUSH6_GPU:?Root must assign the physical GPU after paired timing ends}"
R=/cache/independent_sr_cuda_20261013
exec env CUDA_VISIBLE_DEVICES="$PUSH6_GPU" "$R/env/bin/python" "$R/code/render_cuda_submission.py" \
  --model-dir "$R/submissions/hypir_push6_a7750_ro_wavelet/model_dir" \
  --test-dir "$R/source_payload/assets/input" \
  --val-dir "$R/source_payload/scoring_assets/data/验证集" \
  --out-dir "$R/nr_push6_full_evaluations/push6_a7750_ro_wavelet_all" --cases all
