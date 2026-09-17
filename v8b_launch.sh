#!/bin/bash
# 等教师图拉回完成 -> nanny 挂 v8b -> 全尺度评测
set -u
while [ ! -f /cache/zihaohe/pull.done ]; do sleep 120; done
N=$(cut -d" " -f2 /cache/zihaohe/pull.done)
echo "[launch] 教师图 $N 张 $(date +%H:%M)"
[ "$N" -lt 19900 ] && { echo "[launch] 数量不足"; exit 1; }
cd /cache/zihaohe/csig_isr
bash /cache/zihaohe/nanny.sh 30000 /cache/zihaohe/v8b.log ./venv/bin/python /cache/zihaohe/train_distill.py --init_ckpt /cache/zihaohe/csig_isr/train_out/distill_v5a/best.pth --teacher_dir /cache/zihaohe/distill_data/pdit_teacher --data_mix 0.5 --steps 4000 --bs 4 --lr 8e-6 --val_every 250 --iqa_batch 1 --w_iqa 2.0 --w_maniqa 3.0 --w_clipiqa 2.0 --w_musiq 0.3 --w_niqe 1.0 --w_hf_over 1500 --out /cache/zihaohe/csig_isr/train_out/distill_v8b
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf TORCHDYNAMO_DISABLE=1
for G in 0 1; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
  if [ "$FREE" -gt 25000 ]; then
    CUDA_VISIBLE_DEVICES=$G ./venv/bin/python /cache/zihaohe/eval_ck.py v8b=/cache/zihaohe/csig_isr/train_out/distill_v8b/best.pth >> /cache/zihaohe/v8b.log 2>&1
    break
  fi
done
echo "[launch] 全链完成 $(date +%H:%M)"
