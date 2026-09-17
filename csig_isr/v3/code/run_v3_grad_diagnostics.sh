#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
INIT="${INIT_CKPT:-$CSIG_ROOT/train_out/niqe_long/best.pth}"
GT_DIR="${GT_DIR:-$CSIG_ROOT/AdcSR/testset/DIV2K-Val/HR}"
"$PY" grad_diagnostics.py --ckpt "$INIT" --gt_dir "$GT_DIR" \
  --out_json "$CSIG_ROOT/eval_out/v3_grad_diagnostics.json"
