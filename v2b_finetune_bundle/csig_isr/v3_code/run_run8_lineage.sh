#!/usr/bin/env bash
# 伪影根因是 run12（w_iqa=3.0）+ NIQE 这条血统。run8（w_iqa=1.2）视觉干净但分低。
# 试：从 run8 出发，叠加已验证有效且与 w_iqa 无关的三项——
#   慢衰减 horizon=900（实测 +0.036）、NIQE 目标、HF 约束。
# 保持 w_iqa=1.2 不变，即不走 run12 那条把伪影养出来的路。
source "$(dirname "$0")/_env.sh"
INIT="$CSIG_ROOT/train_out/tinysr8_fid/best.pth"
EXTRA="$CSIG_ROOT/datasets/LSDIR"
COMMON=(--init_ckpt "$INIT" --extra_data "$EXTRA" --w_iqa 1.2 --w_fid 3.0 \
        --steps 450 --val_every 150 --bs 2 --iqa_batch 1 --lr 1e-5 \
        --lr_schedule cosine --lr_horizon 900)
for W in 0.5 2.0; do
  echo "========== run8 血统 w_niqe=$W + hf1500 =========="
  "$PY" train_tinysr.py "${COMMON[@]}" --w_niqe "$W" --w_hf_over 1500 \
    --out "$CSIG_ROOT/train_out/v3_r8_n${W}_hf1500"
done
echo ALLDONE
