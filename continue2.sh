#!/bin/bash
set -u
unset PYTHONPATH PIP_INDEX_URL PIP_TRUSTED_HOST
LAB=/cache/zihaohe/fastwam_lab
export PIP_CACHE_DIR=/cache/zihaohe/pip_cache TMPDIR=/cache/zihaohe/tmp
export HF_HOME=/cache/zihaohe/hf_home HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=/cache/zihaohe/modelscope_cache
cd "$LAB"
AL="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

echo "=== [2] torch (BOTH pinned -> no dependency backtracking) ==="
./venv/bin/pip install -q torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 || { echo "TORCH_FAIL"; exit 1; }
./venv/bin/python -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "=== [3] fastwam + deps ==="
./venv/bin/pip install --no-deps -q -e ./FastWAM
./venv/bin/pip install -q $AL hydra-core==1.3.2 omegaconf==2.3.0 einops safetensors \
  huggingface-hub==0.29.2 modelscope==1.34.0 accelerate==1.12.0 transformers==4.49.0 \
  rich termcolor tqdm regex gitpython jsonlines imageio imageio-ffmpeg boto3 datasets

echo "=== [4] LIBERO patches + deps ==="
touch LIBERO/libero/__init__.py
sed -i 's/torch\.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' LIBERO/libero/libero/benchmark/__init__.py
./venv/bin/pip install -q $AL robosuite==1.4.0 "mujoco==2.3.2" bddl==1.0.1 easydict \
  opencv-python-headless gym==0.25.2 thop matplotlib future cloudpickle
./venv/bin/pip install --force-reinstall --no-deps -q -e ./LIBERO
./venv/bin/python venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py >/dev/null 2>&1
mkdir -p ~/.libero "$LAB/libero_datasets"
LB=$LAB/LIBERO/libero/libero
printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s\nassets: %s/assets\n' \
  "$LB" "$LB" "$LB" "$LAB/libero_datasets" "$LB" > ~/.libero/config.yaml

echo "=== [5] verify ==="
for m in torch fastwam libero robosuite mujoco; do
  ./venv/bin/python -c "import $m" >/dev/null 2>&1 && echo "  OK   $m" || echo "  FAIL $m"
done
MUJOCO_GL=egl ./venv/bin/python -c "
from libero.libero import benchmark
print('  LIBERO n_tasks =', benchmark.get_benchmark_dict()['libero_spatial']().n_tasks)" 2>&1|tail -1

echo "=== [6] checkpoint 12G ==="
cd "$LAB/FastWAM"
"$LAB/venv/bin/huggingface-cli" download yuanty/fastwam \
  libero_optional_idm_2cam224.pt libero_optional_idm_2cam224_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release 2>&1 | tail -2
du -sh ./checkpoints/fastwam_release
echo "=== ALL DONE ==="
