#!/bin/bash
set -u
while ! grep -q FULL_GEN_DONE /cache/zihaohe/vosr2_full.log 2>/dev/null; do sleep 300; done
N=$(ls /cache/zihaohe/distill_data/vosr2_teacher | wc -l)
echo "[v9] 教师图 $N 张 $(date +%H:%M)"
[ "$N" -lt 19900 ] && { echo "[v9] 数量不足,停"; exit 1; }
cd /cache/zihaohe/csig_isr
bash /cache/zihaohe/nanny.sh 30000 /cache/zihaohe/v9.log ./venv/bin/python /cache/zihaohe/train_distill.py --init_ckpt /cache/zihaohe/csig_isr/train_out/distill_v5a/best.pth --teacher_dir /cache/zihaohe/distill_data/vosr2_teacher --data_mix 0.5 --steps 4000 --bs 4 --lr 8e-6 --val_every 250 --iqa_batch 1 --w_iqa 2.0 --w_maniqa 3.0 --w_clipiqa 2.0 --w_musiq 0.3 --w_niqe 1.0 --w_hf_over 1500 --out /cache/zihaohe/csig_isr/train_out/distill_v9
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf TORCHDYNAMO_DISABLE=1
for G in 0 1; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G)
  if [ "$FREE" -gt 25000 ]; then
    CUDA_VISIBLE_DEVICES=$G ./venv/bin/python /cache/zihaohe/eval_ck.py v9=/cache/zihaohe/csig_isr/train_out/distill_v9/best.pth >> /cache/zihaohe/v9.log 2>&1
    break
  fi
done
echo "[v9] 全链完成 $(date +%H:%M)"
