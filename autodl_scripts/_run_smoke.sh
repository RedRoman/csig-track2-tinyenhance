#!/usr/bin/env bash
source /cache/zihaohe/_env_autodl.sh
cd /cache/zihaohe
exec $PY eval_vosr.py vosrcap=/cache/zihaohe/csig_isr/train_out/vosr_cap/best.pth --test 2
