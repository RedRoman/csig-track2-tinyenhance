#!/bin/bash
cd /cache/zihaohe/csig_isr
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf CSIG_ROOT=/cache/zihaohe/csig_isr TINYSR_ROOT=/cache/zihaohe/csig_isr/TinySR PYTHONPATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q FEAT_DONE /cache/zihaohe/feat7.log; do sleep 30; done
C="--arch 64,128,256 --nb 3 --ps 2 --steps 14000 --bs 8 --iqa_batch 1 --lr 1.5e-4 --iqa_start 3000 --gan_start 5000 --val_every 1000"
nohup env CUDA_VISIBLE_DEVICES=0 ./venv/bin/python /cache/zihaohe/train_pix.py $C --feat_dir /cache/zihaohe/distill_data/v8_feat_b7 --w_feat 1.0 --out /cache/zihaohe/csig_isr/train_out/pixFT1 > /cache/zihaohe/pixFT1.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=2 ./venv/bin/python /cache/zihaohe/train_pix.py $C --feat_dir /cache/zihaohe/distill_data/v8_feat_b7 --w_feat 3.0 --out /cache/zihaohe/csig_isr/train_out/pixFT3 > /cache/zihaohe/pixFT3.log 2>&1 &
while ! grep -q FEAT_DONE /cache/zihaohe/feat10.log; do sleep 30; done
nohup env CUDA_VISIBLE_DEVICES=1 ./venv/bin/python /cache/zihaohe/train_pix.py $C --feat_dir /cache/zihaohe/distill_data/v8_feat_b10 --w_feat 1.0 --out /cache/zihaohe/csig_isr/train_out/pixFT10 > /cache/zihaohe/pixFT10.log 2>&1 &
# 对照:同配方同 seed=0 无特征蒸馏 = pixB2 的复现(B2 用的是 seed 0),不用重跑
echo "FT_LAUNCHED $(date +%H:%M)"
