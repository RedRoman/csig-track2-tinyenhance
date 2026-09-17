#!/usr/bin/env bash
# latent 多步细化。起点严格等价于单步（步长 [1,0,...]，已验证逐位恒等 0.000e+00）。
# 回本门槛（p=0.2）：2 步需 ΔScore +0.133，3 步需 +0.224。
source /cache/zihaohe/csig_isr/v3/code/_env.sh
cd /cache/zihaohe/csig_isr/v3/code
INIT=/cache/zihaohe/csig_isr/train_out/run12_iqa30/best.pth
for N in 1 2 3; do
  echo "########## refine_steps=$N ##########"
  "$PY" train_tinysr.py --init_ckpt "$INIT" \
    --extra_data /cache/zihaohe/csig_isr/train_data/lsdir512 \
    --refine_steps $N --step_scale_lr 1e-2 \
    --w_iqa 3.0 --w_fid 3.0 --w_niqe 0.5 \
    --steps 900 --val_every 300 --snapshot_every 300 \
    --bs 2 --iqa_batch 1 --lr 1e-5 --lr_schedule cosine --lr_horizon 1800 \
    --out /cache/zihaohe/csig_isr/train_out/ref$N
  # 每档训完立即同步，实例回收不再丢
  mkdir -p /temp/zihaohe/cache/csig_isr/train_out
  rsync -a --inplace --no-perms --no-owner --no-group \
    /cache/zihaohe/csig_isr/train_out/ref$N \
    /temp/zihaohe/cache/csig_isr/train_out/ 2>/dev/null && echo "  已同步 ref$N 到 /temp"
done
echo ALLDONE
