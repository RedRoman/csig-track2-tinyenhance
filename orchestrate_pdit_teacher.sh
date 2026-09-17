#!/bin/bash
# 自驱动：等 GT 导出 + LQ 传输 -> 传 GT -> AutoDL 双卡分片推理 20k
set -u
while [ ! -f /cache/zihaohe/lq_xfer.done ]; do sleep 60; done
echo "[orch] LQ 传输完成 $(date +%H:%M)"
while ! grep -q "导出 20000" /cache/zihaohe/gt_full.log 2>/dev/null; do sleep 60; done
echo "[orch] GT 导出完成 $(date +%H:%M)"
tar cf - -C /cache/zihaohe/distill_data gt_full | ssh autodl 'tar xf - -C /root/autodl-tmp/zihaohe/distill_full'
echo "[orch] GT 传输完成 $(date +%H:%M)"
ssh autodl 'bash -s' <<'REMOTE'
set -u
cd /root/autodl-tmp/zihaohe/distill_full
n_lq=$(ls lq_mix05 | wc -l); n_gt=$(ls gt_full | wc -l)
echo "[remote] lq=$n_lq gt=$n_gt"
mkdir -p lq_a lq_b gt_a gt_b
for i in $(seq 0 9999); do f=$(printf "%06d.png" $i); ln -sf ../lq_mix05/$f lq_a/$f; ln -sf ../gt_full/$f gt_a/$f; done
for i in $(seq 10000 19999); do f=$(printf "%06d.png" $i); ln -sf ../lq_mix05/$f lq_b/$f; ln -sf ../gt_full/$f gt_b/$f; done
REPO=/root/autodl-tmp/pixeldit_controlnet_continue_200k/PixelDiT
DEL=/root/autodl-tmp/pixeldit_controlnet_csigv2_detail_losses5x/delivery/olddeg_init_current
PROMPT="A clean, natural, high-quality image with sharp, well-defined details, realistic fine textures, clear edges, faithful colors, balanced lighting, and a natural appearance, free from blur, compression artifacts, excessive smoothing, noise, halos, and artificial oversharpening."
cd $REPO
for PART in a b; do
  GPU=2; [ "$PART" = b ] && GPU=3
  nohup env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO/t2i:$REPO" \
    /root/autodl-tmp/songhua/envs/pixeldit-sr/bin/python -m eval_checkpoint_paired_lq \
    --lq-dir /root/autodl-tmp/zihaohe/distill_full/lq_$PART \
    --config "$DEL/train_config.yaml" --checkpoint "$DEL/checkpoint-best-last10" \
    --data /root/autodl-tmp/zihaohe/distill_full/gt_$PART \
    --output /root/autodl-tmp/zihaohe/pdit_teacher_$PART \
    --num-images 10000 --steps 8 --cfg-scale 1.0 --seed 43 \
    --fixed-prompt "$PROMPT" > /root/autodl-tmp/zihaohe/pdit_teacher_$PART.log 2>&1 &
done
echo "[remote] 双卡已启动"
REMOTE
echo "[orch] 全部启动 $(date +%H:%M)"
