#!/usr/bin/env bash
source /cache/zihaohe/_env_autodl.sh
cd /cache/zihaohe
mkdir -p "$CSIG_ROOT/eval_out"
exec $PY eval_vosr.py vosr_iqa1=/cache/zihaohe/csig_isr/train_out/vosr_iqa1.0/best.pth --test 100
