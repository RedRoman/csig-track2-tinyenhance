#!/usr/bin/env bash
set -euo pipefail
# Two independent fixed20 pilot contexts; report scorer S_proxy as S_subset.
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES before running}"
/cache/independent_sr_cuda_20261013/env/bin/python /cache/independent_sr_cuda_20261013/code/score_cuda_stream.py --tool-dir /cache/independent_sr_cuda_20261013/scoring_tool --evaluation nr_stage3_anchor6250_m1_s6500_compiled_pilot=/cache/independent_sr_cuda_20261013/nr_push2_pilot_evaluations/nr_stage3_anchor6250_m1_s6500_compiled_pilot/render --evaluation nr_stage3_anchor6250_m15_s6500_compiled_pilot=/cache/independent_sr_cuda_20261013/nr_push2_pilot_evaluations/nr_stage3_anchor6250_m15_s6500_compiled_pilot/render --out-root /cache/independent_sr_cuda_20261013/nr_push2_pilot_evaluations --test-cases 1,6,7,11,16,18,21,23,25,28,31,36,41,46,51,61,71,81,91,100
