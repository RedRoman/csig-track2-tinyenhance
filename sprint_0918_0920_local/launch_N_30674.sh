#!/bin/bash
# 30674:N1(GPU0) N2(GPU1),各独占一卡
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf; unset PYTHONPATH
PY=./venv/bin/python; L=/cache/zihaohe/pix_runs; T=/cache/zihaohe/csig_isr/train_out; mkdir -p $L
COMMON="--ps 2 --arch 32,64,128 --bs 8 --iqa_batch 1 --workers 6 --steps 60000 --val_every 2000 --warmup 500"
run() { g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g nohup $PY /cache/zihaohe/train_pix.py $COMMON "$@" --out $T/$n > $L/$n.log 2>&1 < /dev/null &
  echo "launched $n on GPU$g pid $!"; }
run 0 pixN1 --nbs 2,2,2 --norm none --lr 2e-4 --min_lr 2e-6
run 1 pixN2 --nbs 1,2,2 --norm gn --init_partial $T/pixL2d/last.pth --lr 1e-4 --min_lr 2e-6 --iqa_start 0 --gan_start 500
