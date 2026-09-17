#!/bin/bash
# 教师生成完 -> 拉回 -> 重命名 -> nanny 挂 v8b -> 全尺度评测
set -u
while :; do
  DONE=$(ssh autodl 'a=0; b=0; test -f /root/autodl-tmp/zihaohe/pdit_teacher_a/metrics.json && ! pgrep -f "pdit_teacher_a" >/dev/null && a=1; test -f /root/autodl-tmp/zihaohe/pdit_teacher_b/metrics.json && ! pgrep -f "pdit_teacher_b" >/dev/null && b=1; echo $a$b' 2>/dev/null | tail -1)
  [ "$DONE" = "11" ] && break
  sleep 600
done
echo "[relay] 教师生成完成 $(date +%H:%M)"
mkdir -p /cache/zihaohe/distill_data/pdit_teacher
for PART in a b; do
  ssh autodl "cd /root/autodl-tmp/zihaohe/pdit_teacher_$PART/samples && tar cf - *_sr.png" | tar xf - -C /cache/zihaohe/distill_data/pdit_teacher
done
cd /cache/zihaohe/distill_data/pdit_teacher
for f in *_sr.png; do idx=$(echo "$f" | cut -d_ -f2); mv -n "$f" "$idx.png"; done
N=$(ls | wc -l); echo "[relay] 教师图 $N 张已就位 $(date +%H:%M)"
[ "$N" -lt 19900 ] && { echo "[relay] 数量不足，停"; exit 1; }
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
echo "[relay] 全链完成 $(date +%H:%M)"
