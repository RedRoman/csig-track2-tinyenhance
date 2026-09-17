#!/bin/bash
cd /cache/zihaohe/csig_isr
export TORCHDYNAMO_DISABLE=
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf PYTHONPATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./venv/bin/python /cache/zihaohe/check_v10.py /cache/zihaohe/csig_isr/submit_v10/model_dir
grep -q CHECK_V10_OK /cache/zihaohe/v10chain.log || exit 1
./venv/bin/python /cache/zihaohe/render_v10.py
