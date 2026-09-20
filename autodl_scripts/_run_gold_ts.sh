#!/usr/bin/env bash
source /cache/zihaohe/_env_autodl.sh
cd /cache/zihaohe
export CUDA_VISIBLE_DEVICES=1
mkdir -p "$CSIG_ROOT/eval_out"
exec $PY eval_ck.py comboexact=/cache/zihaohe/csig_isr/train_out/combo_exact/best.pth --test 100
