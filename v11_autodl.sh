#!/bin/bash
ssh autodl 'mkdir -p /root/autodl-tmp/zihaohe/csig_isr/submit_v11 && rm -rf /root/autodl-tmp/zihaohe/csig_isr/submit_v11/model_dir'
scp -q -r /cache/zihaohe/csig_isr/submit_v11/model_dir autodl:/root/autodl-tmp/zihaohe/csig_isr/submit_v11/ && echo "[xfer] model_dir 传输完成 $(date +%H:%M)"
scp -q /cache/zihaohe/check_v11.py /cache/zihaohe/render_v11.py autodl:/root/autodl-tmp/zihaohe/
ssh autodl 'bash -s' <<'REMOTE'
cd /cache/zihaohe/csig_isr
export TORCHDYNAMO_DISABLE= HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/hf PYTHONPATH= CSIG_ROOT=/cache/zihaohe/csig_isr TINYSR_ROOT=/cache/zihaohe/csig_isr/TinySR
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=== A. 缓存路径(同py/torch,异GPU) ==="
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python /cache/zihaohe/check_v11.py /cache/zihaohe/csig_isr/submit_v11/model_dir 2>&1 | grep -E "init|污染|median|4K|CHECK|Error|fallback|Assert"
rm -rf /cache/zihaohe/v11cold && cp -r /cache/zihaohe/csig_isr/submit_v11/model_dir /cache/zihaohe/v11cold && rm -rf /cache/zihaohe/v11cold/_work && sed -i 's|same_env = (sys.version_info\[:2\] == (3, 10)) and torch.__version__.startswith("2.5")|same_env = False|' /cache/zihaohe/v11cold/runner.py
echo "=== B. 冷路径(reduce-overhead 冷编译) ==="
CUDA_VISIBLE_DEVICES=1 ./venv/bin/python /cache/zihaohe/check_v11.py /cache/zihaohe/v11cold 2>&1 | grep -E "init|污染|median|4K|CHECK|Error|fallback|Assert"
echo "=== C. 渲染 ==="
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python /cache/zihaohe/render_v11.py 2>&1 | grep -E "img|DONE|Error"
rm -rf /cache/zihaohe/csig_isr/submit_v11/model_dir/_work
cd /cache/zihaohe/csig_isr/submit_v11 && find model_dir -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; zip -q -r "赛题二_TinyEnhance_干就完了_18020768360.zip" output_dir model_dir -x "*.pyc" && echo "ZIP_MD5 $(md5sum *.zip | cut -c1-8) files=$(unzip -l *.zip | tail -1 | awk '{print $2}')"
REMOTE
echo "AUTODL_CHAIN_DONE $(date +%H:%M)"
