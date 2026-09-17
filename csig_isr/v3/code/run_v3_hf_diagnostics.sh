#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
RUN8="${RUN8:-$CSIG_ROOT/train_out/tinysr8_fid/best.pth}"
W05="${W05:-$CSIG_ROOT/train_out/niqe_w0.5/best.pth}"
NLONG="${NLONG:-$CSIG_ROOT/train_out/niqe_long/best.pth}"
"$PY" hf_diagnostics.py \
  --candidate run8 "$RUN8" 1.0 \
  --candidate niqe_w0.5 "$W05" 1.0 \
  --candidate niqe_long "$NLONG" 1.0 \
  --topk_maps 5 --out_json "$CSIG_ROOT/eval_out/v3_hf_diagnostics.json"
