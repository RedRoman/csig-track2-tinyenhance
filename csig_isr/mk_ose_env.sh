#!/bin/bash
# 建 OSEDiff 专用环境（diffusers 0.25 + transformers 4.28，与主环境隔离）
set -e
unset PYTHONPATH
export PIP_CACHE_DIR=/cache/zihaohe/.pipcache
export TMPDIR=/cache/zihaohe/tmp
cd /cache/zihaohe/csig_isr

rm -rf venv_ose
/home/ma-user/.local/bin/virtualenv venv_ose --python=python3.10

echo "=== 装 torch (走 pip cache) ==="
venv_ose/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2

echo "=== 装 OSEDiff 依赖 ==="
venv_ose/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "diffusers==0.25.0" "transformers==4.28.1" "peft==0.9.0" \
    "huggingface_hub==0.25.0" "numpy==1.26.4" "setuptools<81" \
    einops "open-clip-torch==2.20.0" timm fairscale loralib pyyaml \
    opencv-python safetensors accelerate 2>&1 | tail -3

echo "=== 验证 ==="
venv_ose/bin/python -c "
import torch, diffusers, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)
"
echo "OSE_ENV_DONE"
