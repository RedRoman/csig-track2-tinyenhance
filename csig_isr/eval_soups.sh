#!/bin/bash
unset PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CSIG_ROOT=/cache/zihaohe/csig_isr TINYSR_ROOT=/cache/zihaohe/csig_isr/TinySR
cd /cache/zihaohe/csig_isr
for lam in 0.00 0.30 0.50 0.70 1.00 1.15 1.30; do
  echo "########## soup lambda=$lam (alpha=1.20) ##########"
  venv/bin/python v2/code/score_eval.py \
    --ckpt /cache/zihaohe/csig_isr/train_out/v2_soup/soup_${lam}.pth \
    --alpha_mode fixed --alpha 1.20 \
    --out_json /cache/zihaohe/csig_isr/eval_out/v2_soup_${lam}.json 2>&1 \
  | grep -E "^score=|^case=|^tier=|Error|Traceback"
done
echo "SOUP_EVAL_DONE"
