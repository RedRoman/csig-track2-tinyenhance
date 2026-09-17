#!/bin/bash
cd /cache/zihaohe/csig_isr
export TORCHDYNAMO_DISABLE=
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf PYTHONPATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./venv/bin/python /cache/zihaohe/check_v10.py /cache/zihaohe/csig_isr/submit_v10b/model_dir
grep -q CHECK_V10_OK /cache/zihaohe/v10bchain.log || exit 1
./venv/bin/python /cache/zihaohe/render_v10b.py
grep -q RENDER_V10B_DONE /cache/zihaohe/v10bchain.log || exit 1
cd /cache/zihaohe/csig_isr/submit_v10b && find model_dir -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; zip -q -r "赛题二_TinyEnhance_干就完了_18020768360.zip" output_dir model_dir -x "*.pyc" && echo "ZIP_MD5 $(md5sum *.zip | cut -c1-8) files=$(unzip -l *.zip | tail -1 | awk '{print $2}')"
