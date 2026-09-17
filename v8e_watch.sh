#!/bin/bash
while ! grep -qE "ALLDONE|NANNY_GAVEUP" /cache/zihaohe/v8e.log 2>/dev/null; do sleep 300; done
grep -q NANNY_GAVEUP /cache/zihaohe/v8e.log && exit 1
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for TRY in $(seq 1 24); do
  for G in 0 1; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
    if [ "$FREE" -gt 15000 ]; then
      CUDA_VISIBLE_DEVICES=$G ./venv/bin/python /cache/zihaohe/eval_ck.py v8e=/cache/zihaohe/csig_isr/train_out/distill_v8e/best.pth >> /cache/zihaohe/v8e.log 2>&1
      exit 0
    fi
  done
  sleep 600
done
