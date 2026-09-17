#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
INIT="${INIT_CKPT:-$CSIG_ROOT/train_out/niqe_long/best.pth}"
EXTRA="${EXTRA_DATA:-$CSIG_ROOT/datasets/LSDIR}"
COMMON=(--init_ckpt "$INIT" --extra_data "$EXTRA" --steps 300 --val_every 50 \
        --bs 2 --iqa_batch 1 --lr 1e-6 --lr_schedule cosine --lr_horizon 900 --w_fid 3.0)

# Mild: keep more NR pressure, repair fidelity gently.
"$PY" train_tinysr.py "${COMMON[@]}" --w_iqa 1.5 --w_niqe 0.5 \
  --w_lpips 1.25 --w_dists 1.75 --out "$CSIG_ROOT/train_out/v3_repair_mild"

# Balanced: intended default candidate.
"$PY" train_tinysr.py "${COMMON[@]}" --w_iqa 1.0 --w_niqe 0.25 \
  --w_lpips 1.5 --w_dists 2.0 --out "$CSIG_ROOT/train_out/v3_repair_balanced"

# Strong: diagnostic endpoint; may sacrifice proxy score for FR recovery.
"$PY" train_tinysr.py "${COMMON[@]}" --w_iqa 0.5 --w_niqe 0.0 \
  --w_lpips 2.0 --w_dists 2.5 --out "$CSIG_ROOT/train_out/v3_repair_strong"
