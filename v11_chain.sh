#!/bin/bash
cd /cache/zihaohe/csig_isr
export TORCHDYNAMO_DISABLE=
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf PYTHONPATH= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== A. 缓存路径(同环境) ==="
./venv/bin/python /cache/zihaohe/check_v11.py /cache/zihaohe/csig_isr/submit_v11/model_dir
echo "=== B. 冷路径(模拟异环境, reduce-overhead 冷编译) ==="
./venv/bin/python /cache/zihaohe/check_v11.py /tmp/v11cold/model_dir
grep -c CHECK_V11_OK /cache/zihaohe/v11chain.log | grep -q 2 || exit 1
echo "=== C. 渲染 ==="
./venv/bin/python /cache/zihaohe/render_v11.py
grep -q RENDER_V11_DONE /cache/zihaohe/v11chain.log || exit 1
rm -rf /cache/zihaohe/csig_isr/submit_v11/model_dir/_work
cd /cache/zihaohe/csig_isr/submit_v11 && find model_dir -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; zip -q -r "赛题二_TinyEnhance_干就完了_18020768360.zip" output_dir model_dir -x "*.pyc" && echo "ZIP_MD5 $(md5sum *.zip | cut -c1-8) files=$(unzip -l *.zip | tail -1 | awk '{print $2}')"
