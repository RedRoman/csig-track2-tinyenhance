#!/usr/bin/env bash
cd /cache/zihaohe/VOSR
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/cache/zihaohe/csig_isr/hf
export TORCH_HOME=/cache/zihaohe/VOSR_weights/torch_cache
PY=/cache/zihaohe/csig_isr/venv/bin/python

echo "=== 补依赖 ==="
$PY -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple omegaconf einops timm 2>&1 | tail -2

echo "=== 备测试图 ==="
mkdir -p /cache/zihaohe/vosr_test/in
$PY -c "
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
im=Image.open('/cache/zihaohe/csig_isr/data/赛题二/验证集/case1_lq.jpg').convert('RGB')
w,h=im.size
im.crop((w//2-256,h//2-256,w//2+256,h//2+256)).save('/cache/zihaohe/vosr_test/in/probe.png')
print('ok', w, h)
"

echo "=== 跑推理 ==="
$PY inference_vosr_onestep.py \
  -c /cache/zihaohe/VOSR_weights/VOSR_0.5B_os \
  -i /cache/zihaohe/vosr_test/in \
  -o /cache/zihaohe/vosr_test/out \
  -u 1 --infer_steps 1 2>&1 | tail -25
echo VOSR_PROBE_DONE
