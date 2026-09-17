#!/usr/bin/env bash
# 把已验证的最优训练配置（w_niqe=2.0 + horizon=900）与 HF 约束合起来，
# 让约束全程塑形，而不是事后 repair。
# w_hf_over 按主训练权重标定：主梯度=1.0*lpips+1.5*dists=4.71，
#   500 -> 约 7%，1500 -> 约 21%
source "$(dirname "$0")/_env.sh"
INIT="$CSIG_ROOT/train_out/run12_iqa30/best.pth"
EXTRA="$CSIG_ROOT/datasets/LSDIR"
COMMON=(--init_ckpt "$INIT" --extra_data "$EXTRA" \
        --w_iqa 3.0 --w_fid 3.0 --w_niqe 2.0 \
        --steps 450 --val_every 150 --bs 2 --iqa_batch 1 --lr 1e-5 \
        --lr_schedule cosine --lr_horizon 900)
for W in 500 1500; do
  echo "========== combo w_hf_over=$W =========="
  "$PY" train_tinysr.py "${COMMON[@]}" --w_hf_over "$W" \
    --out "$CSIG_ROOT/train_out/v3_combo_hf$W"
done
echo ALLDONE
