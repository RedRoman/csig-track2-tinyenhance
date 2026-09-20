#!/bin/bash
# AutoDL:N3(GPU0) N4(GPU1) N5(GPU2) N3b(GPU3);数据盘已满,ckpt 写到系统盘 /root/pix_out(每条仅 best+last ~17MB)
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf; unset PYTHONPATH
PY=./venv/bin/python; L=/root/pix_out/logs; T=/root/pix_out; mkdir -p $L
COMMON="--ps 2 --arch 32,64,128 --bs 8 --iqa_batch 1 --workers 6 --steps 60000 --val_every 2000 --warmup 500"
run() { g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g nohup $PY /cache/zihaohe/train_pix.py $COMMON "$@" --out $T/$n > $L/$n.log 2>&1 < /dev/null &
  echo "launched $n on GPU$g pid $!"; }
run 0 pixN3 --nbs 1,2,3 --norm none --lr 2e-4 --min_lr 2e-6
run 1 pixN4 --nbs 1,2,2 --norm none --lr 2e-4 --min_lr 2e-6
run 2 pixN5 --nbs 1,1,2 --norm none --lr 2e-4 --min_lr 2e-6
run 3 pixN3b --nbs 1,2,3 --norm none --lr 2e-4 --min_lr 2e-6 --seed 1
