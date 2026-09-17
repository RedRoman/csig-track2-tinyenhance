#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
# Run only after hf_diagnostics.py + grad_diagnostics.py. Override W_HF_LIST after
# calibrating the HF gradient magnitude; defaults are intentionally conservative.
INIT="${INIT_CKPT:-$CSIG_ROOT/train_out/niqe_long/best.pth}"
EXTRA="${EXTRA_DATA:-$CSIG_ROOT/datasets/LSDIR}"
W_HF_LIST="${W_HF_LIST:-0.25 1.0 4.0}"
for W in $W_HF_LIST; do
  echo "========== w_hf_over=$W =========="
  "$PY" train_tinysr.py --init_ckpt "$INIT" --extra_data "$EXTRA" \
    --steps 300 --val_every 50 --bs 2 --iqa_batch 1 --lr 1e-6 --lr_horizon 900 \
    --w_iqa 1.0 --w_niqe 0.25 --w_fid 3.0 --w_lpips 1.5 --w_dists 2.0 \
    --w_hf_over "$W" --out "$CSIG_ROOT/train_out/v3_hf_w$W"
done
