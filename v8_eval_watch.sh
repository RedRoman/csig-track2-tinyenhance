#!/bin/bash
while ! grep -q "ALLDONE\|NANNY_GAVEUP" /cache/zihaohe/v8.log 2>/dev/null; do sleep 120; done
grep -q NANNY_GAVEUP /cache/zihaohe/v8.log && exit 1
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf TORCHDYNAMO_DISABLE=1
for G in 0 1; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
  if [ "$FREE" -gt 25000 ]; then
    CUDA_VISIBLE_DEVICES=$G ./venv/bin/python /cache/zihaohe/eval_ck.py v8=/cache/zihaohe/csig_isr/train_out/distill_v8/best.pth >> /cache/zihaohe/v8.log 2>&1
    break
  fi
done
