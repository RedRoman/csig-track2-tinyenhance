#!/usr/bin/env bash
# 在 3 张全分辨率验证图上跑 VOSR，供我们自己的评测器打分
cd /cache/zihaohe/VOSR
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/cache/zihaohe/csig_isr/hf
export TORCH_HOME=/cache/zihaohe/VOSR_weights/torch_cache
PY=/cache/zihaohe/csig_isr/venv/bin/python
V=/cache/zihaohe/csig_isr/data/赛题二/验证集
IN=/cache/zihaohe/vosr_val/in
mkdir -p $IN
cp $V/case1_lq.jpg $V/case2_lq.jpg $V/case3_lq.jpg $IN/ 2>/dev/null
ls $IN
for CFG in 0.5; do
  echo "########## cfg_scale=$CFG ##########"
  $PY inference_vosr_onestep.py \
    -c /cache/zihaohe/VOSR_weights/VOSR_0.5B_os \
    -i $IN -o /cache/zihaohe/vosr_val/out_cfg$CFG \
    -u 1 --infer_steps 1 --tile_size 512 --tile_overlap 256 \
    --align_method wavelet 2>&1 | tail -6
done
echo VOSR_VAL_DONE
