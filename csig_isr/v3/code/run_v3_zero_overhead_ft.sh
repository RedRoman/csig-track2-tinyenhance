#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
# These are clean ablations after the v2 AMP bug fix. INIT_CKPT should be the
# best safe/repair candidate chosen from P1, not necessarily raw niqe_long.
INIT="${INIT_CKPT:-$CSIG_ROOT/train_out/niqe_long/best.pth}"
EXTRA="${EXTRA_DATA:-$CSIG_ROOT/datasets/LSDIR}"
COMMON=(--init_ckpt "$INIT" --extra_data "$EXTRA" --steps 300 --val_every 50 \
        --bs 2 --iqa_batch 1 --lr 2e-6 --dense_lr 5e-7 --lr_horizon 900 \
        --w_iqa 1.0 --w_niqe 0.25 --w_fid 3.0 --w_lpips 1.5 --w_dists 2.0)

# Decoder-only: zero extra inference ops; tests whether the tiny VAE can represent
# a better fidelity/naturalness trade-off without changing the Transformer.
"$PY" train_tinysr.py "${COMMON[@]}" --freeze_lora --train_vae_decoder \
  --out "$CSIG_ROOT/train_out/v3_decoder_only"

# Last-2 DiT blocks only: also zero extra inference ops.
"$PY" train_tinysr.py "${COMMON[@]}" --freeze_lora --unfreeze_last_n 2 \
  --out "$CSIG_ROOT/train_out/v3_dense_last2"
