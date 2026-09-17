#!/bin/bash
# 用法: nanny.sh <需要的显存MB> <日志> <命令...>
NEED=$1; LOG=$2; shift 2
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf CSIG_ROOT=/cache/zihaohe/csig_isr TINYSR_ROOT=/cache/zihaohe/csig_isr/TinySR PYTHONPATH= TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for TRY in $(seq 1 48); do
  for G in 0 1; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
    if [ "$FREE" -gt "$NEED" ]; then
      echo "[nanny] try=$TRY gpu=$G free=${FREE}MB 启动" >> $LOG
      CUDA_VISIBLE_DEVICES=$G "$@" >> $LOG 2>&1
      if ! grep -q "OutOfMemoryError" <(tail -30 $LOG); then
        echo "ALLDONE" >> $LOG; exit 0
      fi
      echo "[nanny] OOM，重排队" >> $LOG
    fi
  done
  sleep 600
done
echo "NANNY_GAVEUP" >> $LOG
