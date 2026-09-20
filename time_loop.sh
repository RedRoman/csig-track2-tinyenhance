#!/bin/bash
# 独占卡按序单模型计时;L2d 首尾各一次校漂移。用法: time_loop.sh GPU OUTLOG 名=ckpt ...
G=$1; O=$2; shift 2
cd /cache/zihaohe/csig_isr; export PYTHONPATH= HF_ENDPOINT=https://hf-mirror.com TORCHDYNAMO_DISABLE=
: > $O
for spec in "$@" "L2d_again=/cache/zihaohe/csig_isr/train_out/pixL2d/last.pth"; do
  n=${spec%%=*}; ck=${spec#*=}
  CUDA_VISIBLE_DEVICES=$G ./venv/bin/python /cache/zihaohe/time_one.py $n $ck 2>&1 | grep -E "^TIME|Error|Traceback" >> $O
done
echo LOOP_DONE >> $O
