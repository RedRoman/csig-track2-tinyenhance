#!/bin/bash
# $1 = GPU id, $2 = keep_layers, $3 = out 名
set -u
G=$1; KEEP=$2; NAME=$3
while ! grep -q V8GEN_DONE /cache/zihaohe/v8gen.log 2>/dev/null; do sleep 60; done
N=$(ls /cache/zihaohe/distill_data/v8_teacher | wc -l)
echo "[$NAME] 教师 $N 张 $(date +%H:%M)"
[ "$N" -lt 19900 ] && { echo "[$NAME] 教师不足"; exit 1; }
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf CSIG_ROOT=/cache/zihaohe/csig_isr TINYSR_ROOT=/cache/zihaohe/csig_isr/TinySR PYTHONPATH= TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$G ./venv/bin/python /cache/zihaohe/train_distill.py \
  --init_ckpt /cache/zihaohe/csig_isr/train_out/distill_v8/best.pth \
  --keep_layers "$KEEP" --dense_lr 1e-5 \
  --teacher_dir /cache/zihaohe/distill_data/v8_teacher --data_mix 0.5 \
  --steps 6000 --bs 4 --lr 8e-6 --val_every 250 --iqa_batch 1 \
  --w_iqa 2.0 --w_maniqa 3.0 --w_clipiqa 2.0 --w_musiq 0.3 --w_niqe 1.0 \
  --w_hf_over 1500 --w_gan 0.05 --gan_start 500 --adaptive_alpha \
  --out /cache/zihaohe/csig_isr/train_out/$NAME
echo "[$NAME] 训练结束 $(date +%H:%M)"
