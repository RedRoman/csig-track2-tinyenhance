#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
INIT="${INIT_CKPT:-$CSIG_ROOT/train_out/run12_iqa30/best.pth}"
EXTRA="${EXTRA_DATA:-$CSIG_ROOT/datasets/LSDIR}"
COMMON=(--init_ckpt "$INIT" --w_iqa 3.0 --w_fid 3.0 --w_niqe 0.5 \
        --extra_data "$EXTRA" --steps 600 --val_every 150 --bs 2 --iqa_batch 1 --lr 1e-5)

for H in 900 1800; do
  echo "========== safe NIQE 0.5, horizon=$H =========="
  "$PY" train_tinysr.py "${COMMON[@]}" --lr_schedule cosine --lr_horizon "$H" \
    --out "$CSIG_ROOT/train_out/v3_safe_niqe_h$H"
done

echo "========== safe NIQE 0.5, constant LR =========="
"$PY" train_tinysr.py "${COMMON[@]}" --lr_schedule constant \
  --out "$CSIG_ROOT/train_out/v3_safe_niqe_constant"
