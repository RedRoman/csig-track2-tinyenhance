#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
OUT="$CSIG_ROOT/eval_out/v3_baselines"
mkdir -p "$OUT"
RUN8="${RUN8:-$CSIG_ROOT/train_out/tinysr8_fid/best.pth}"
W05="${W05:-$CSIG_ROOT/train_out/niqe_w0.5/best.pth}"
NLONG="${NLONG:-$CSIG_ROOT/train_out/niqe_long/best.pth}"

for SPEC in "run8:$RUN8" "niqe_w0.5:$W05" "niqe_long:$NLONG"; do
  NAME="${SPEC%%:*}"; CKPT="${SPEC#*:}"
  echo "========== $NAME =========="
  "$PY" score_eval.py --ckpt "$CKPT" --alpha_mode fixed --alpha 1.0 --repeats 3 \
    --out_json "$OUT/$NAME.json"
done
