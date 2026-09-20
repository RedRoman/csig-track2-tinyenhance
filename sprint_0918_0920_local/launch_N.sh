#!/bin/bash
# 30674 上启动 4 条 v2 架构变体(2 条/卡)。用法: launch_N.sh [smoke]
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf; unset PYTHONPATH
PY=./venv/bin/python; L=/cache/zihaohe/pix_runs; mkdir -p $L
if [ "$1" = "smoke" ]; then EXTRA="--steps 40 --val_every 40 --val_crops 2 --warmup 5"; SUF=_smoke; else EXTRA="--steps 60000 --val_every 2000 --warmup 500"; SUF=; fi
COMMON="--ps 2 --arch 32,64,128 --bs 8 --iqa_batch 1 --workers 6 $EXTRA"
run() { # gpu name args...
  g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g nohup $PY /cache/zihaohe/train_pix.py $COMMON "$@" --out train_out/$n$SUF > $L/$n$SUF.log 2>&1 < /dev/null &
  echo "launched $n on GPU$g pid $!"
}
run 0 pixN1 --nbs 2,2,2 --norm none --lr 2e-4 --min_lr 2e-6
run 0 pixN2 --nbs 1,2,2 --norm gn --init_partial train_out/pixL2d/last.pth --lr 1e-4 --min_lr 2e-6 --iqa_start 0 --gan_start 500
run 1 pixN3 --nbs 1,2,3 --norm none --lr 2e-4 --min_lr 2e-6
run 1 pixN4 --nbs 1,2,2 --norm none --lr 2e-4 --min_lr 2e-6
